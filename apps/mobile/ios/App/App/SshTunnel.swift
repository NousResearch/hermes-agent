import Citadel
import Crypto
import Foundation
import NIO
import NIOPosix
import NIOSSH

struct RemoteSessionIdentity: Codable, Sendable {
    let ownershipID: String
    let nonce: String
}

struct SshTunnelConfiguration: Sendable {
    let host: String
    let port: Int
    let username: String
    let privateKey: String
    let hostKey: String
    let previousSession: RemoteSessionIdentity?
}

private struct RemoteBackend: Sendable {
    let port: Int
    let pid: Int
    let ownershipID: String
    let nonce: String
    let token: String

    var identity: RemoteSessionIdentity {
        RemoteSessionIdentity(ownershipID: ownershipID, nonce: nonce)
    }
}

enum SshTunnelError: LocalizedError {
    case alreadyStarted
    case notStarted
    case listenerHasNoPort

    var errorDescription: String? {
        switch self {
        case .alreadyStarted:
            return "The SSH tunnel is already running."
        case .notStarted:
            return "The SSH tunnel is not running."
        case .listenerHasNoPort:
            return "The local tunnel listener did not expose a port."
        }
    }
}

private final class LocalToSSHHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    private var remoteChannel: Channel?
    private var pendingData: [ByteBuffer] = []

    func setRemoteChannel(_ channel: Channel) {
        remoteChannel = channel
        for data in pendingData {
            channel.write(data, promise: nil)
        }
        pendingData.removeAll(keepingCapacity: false)
        channel.flush()
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let buffer = self.unwrapInboundIn(data)
        guard let remoteChannel else {
            pendingData.append(buffer)
            return
        }
        remoteChannel.writeAndFlush(buffer, promise: nil)
    }

    func channelReadComplete(context: ChannelHandlerContext) {
        remoteChannel?.flush()
        context.flush()
    }

    func channelInactive(context: ChannelHandlerContext) {
        remoteChannel?.close(promise: nil)
        context.fireChannelInactive()
    }
}

private final class SSHToLocalHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    private weak var localChannel: Channel?

    init(localChannel: Channel) {
        self.localChannel = localChannel
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard let localChannel else {
            context.close(promise: nil)
            return
        }
        localChannel.writeAndFlush(self.unwrapInboundIn(data), promise: nil)
    }

    func channelReadComplete(context: ChannelHandlerContext) {
        localChannel?.flush()
        context.flush()
    }

    func channelInactive(context: ChannelHandlerContext) {
        localChannel?.close(promise: nil)
        context.fireChannelInactive()
    }
}

private func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private func pythonQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "\\'") + "'"
}

private func remoteHermesHomePath(_ suffix: String) -> String {
    "\"${HERMES_HOME:-$HOME/.hermes}/desktop-ssh/\(suffix)\""
}

actor SshTunnel {
    private var client: SSHClient?
    private var listener: Channel?
    private var eventLoopGroup: MultiThreadedEventLoopGroup?
    private var remoteBackend: RemoteBackend?

    var isRunning: Bool {
        listener?.isActive == true && client?.isConnected == true
    }

    var localPort: UInt16? {
        guard isRunning, let port = listener?.localAddress?.port else {
            return nil
        }
        return UInt16(port)
    }

    var sessionToken: String? {
        remoteBackend?.token
    }

    var sessionIdentity: RemoteSessionIdentity? {
        remoteBackend?.identity
    }

    func start(_ configuration: SshTunnelConfiguration) async throws -> UInt16 {
        guard listener == nil else {
            throw SshTunnelError.alreadyStarted
        }

        let privateKey = try Curve25519.Signing.PrivateKey(sshEd25519: configuration.privateKey)
        let hostKey = try NIOSSHPublicKey(openSSHPublicKey: configuration.hostKey)
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        eventLoopGroup = group

        var startedBackend: RemoteBackend?
        do {
            var settings = SSHClientSettings(
                host: configuration.host,
                port: configuration.port,
                authenticationMethod: {
                    .ed25519(
                        username: configuration.username,
                        privateKey: privateKey
                    )
                },
                hostKeyValidator: .trustedKeys([hostKey])
            )
            settings.group = group
            settings.connectTimeout = TimeAmount.seconds(15)

            let sshClient = try await SSHClient.connect(to: settings)
            client = sshClient
            if let previousSession = configuration.previousSession {
                await stopRemoteSession(previousSession, using: sshClient)
            }
            let backend = try await startRemoteBackend(using: sshClient)
            startedBackend = backend
            let bootstrap = ServerBootstrap(group: group)
                .serverChannelOption(ChannelOptions.backlog, value: 16)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelInitializer { channel in
                    let handler = LocalToSSHHandler()
                    return channel.pipeline.addHandler(handler).map {
                        Task {
                            do {
                                let originator = try SocketAddress(ipAddress: "127.0.0.1", port: 0)
                                let remote = try await sshClient.createDirectTCPIPChannel(
                                    using: SSHChannelType.DirectTCPIP(
                                        targetHost: "127.0.0.1",
                                        targetPort: backend.port,
                                        originatorAddress: originator
                                    )
                                ) { remoteChannel in
                                    remoteChannel.pipeline.addHandler(
                                        SSHToLocalHandler(localChannel: channel)
                                    )
                                }
                                channel.eventLoop.execute {
                                    handler.setRemoteChannel(remote)
                                }
                            } catch {
                                channel.close(promise: nil)
                            }
                        }
                    }
                }
                .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelOption(ChannelOptions.allowRemoteHalfClosure, value: true)

            let localListener = try await bootstrap.bind(host: "127.0.0.1", port: 0).get()
            guard let localPort = localListener.localAddress?.port else {
                try? await localListener.close()
                try? await sshClient.close()
                try? await group.shutdownGracefully()
                eventLoopGroup = nil
                throw SshTunnelError.listenerHasNoPort
            }

            client = sshClient
            listener = localListener
            remoteBackend = backend
            startedBackend = nil
            return UInt16(localPort)
        } catch {
            if let backend = remoteBackend ?? startedBackend {
                await stopRemoteBackend(backend, using: client)
            }
            remoteBackend = nil
            try? await client?.close()
            try? await group.shutdownGracefully()
            eventLoopGroup = nil
            throw error
        }
    }

    private func startRemoteBackend(using sshClient: SSHClient) async throws -> RemoteBackend {
        let hermesOutput = try await sshClient.executeCommand(
            "for p in \"$HOME/.local/bin/hermes\" \"${HERMES_HOME:-$HOME/.hermes}/hermes-agent/venv/bin/hermes\"; do [ -x \"$p\" ] && printf '%s' \"$p\" && exit 0; done; command -v hermes"
        )
        let hermesPath = String(buffer: hermesOutput).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !hermesPath.isEmpty else {
            throw NSError(domain: "SshTunnel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hermes executable was not found on the remote host."])
        }

        let ownershipID = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let nonce = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(16))
        let tokenSource = "import os,secrets; h=os.environ.get('HERMES_HOME') or os.path.expanduser('~/.hermes'); p=os.path.join(h,'desktop-ssh',\(pythonQuote(ownershipID)),\(pythonQuote("\(nonce).token"))); os.makedirs(os.path.dirname(p),mode=0o700,exist_ok=True); f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(f,secrets.token_hex(32).encode()); os.close(f)"

        _ = try await sshClient.executeCommand("python3 -c \(shellQuote(tokenSource)) >/dev/null 2>&1")
        let tokenOutput = try await sshClient.executeCommand("cat \(remoteHermesHomePath("\(ownershipID)/\(nonce).token"))")
        let token = String(buffer: tokenOutput).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "SshTunnel", code: 2, userInfo: [NSLocalizedDescriptionKey: "The remote Hermes session token was empty."])
        }

        let launch = "env HERMES_DESKTOP=1 \(shellQuote(hermesPath)) serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file \(remoteHermesHomePath("\(ownershipID)/\(nonce).token")) --ssh-owner-nonce \(nonce) </dev/null >> \(remoteHermesHomePath("\(ownershipID)/\(nonce).log")) 2>&1 & echo $!"
        let command = "mkdir -p \(remoteHermesHomePath("")) && \"$(command -v setsid || echo nohup)\" sh -c \(shellQuote(launch))"
        let pidOutput = try await sshClient.executeCommand(command)
        guard let pid = Int(String(buffer: pidOutput).trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw NSError(domain: "SshTunnel", code: 3, userInfo: [NSLocalizedDescriptionKey: "The remote Hermes backend did not return a process ID."])
        }

        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline {
            do {
                let readyOutput = try await sshClient.executeCommand("grep -oE 'HERMES_(BACKEND|DASHBOARD)_READY port=[0-9]+' \(remoteHermesHomePath("\(ownershipID)/\(nonce).log")) 2>/dev/null | tail -n 1 | sed 's/.*port=//'")
                if let port = Int(String(buffer: readyOutput).trimmingCharacters(in: .whitespacesAndNewlines)), port > 0 {
                    return RemoteBackend(port: port, pid: pid, ownershipID: ownershipID, nonce: nonce, token: token)
                }
            } catch {
                // The remote log is not ready yet; continue polling.
            }
            try await Task.sleep(nanoseconds: 250_000_000)
        }

        await stopRemoteBackend(RemoteBackend(port: 0, pid: pid, ownershipID: ownershipID, nonce: nonce, token: token), using: sshClient)
        throw NSError(domain: "SshTunnel", code: 4, userInfo: [NSLocalizedDescriptionKey: "The remote Hermes backend did not announce a ready port."])
    }

    private func stopRemoteBackend(_ backend: RemoteBackend, using sshClient: SSHClient?) async {
        guard let sshClient else { return }
        await stopRemoteSession(backend.identity, pid: backend.pid, using: sshClient)
    }

    private func stopRemoteSession(_ identity: RemoteSessionIdentity, using sshClient: SSHClient) async {
        await stopRemoteSession(identity, pid: nil, using: sshClient)
    }

    private func stopRemoteSession(_ identity: RemoteSessionIdentity, pid: Int?, using sshClient: SSHClient) async {
        let tokenPath = remoteHermesHomePath("\(identity.ownershipID)/\(identity.nonce).token")
        let logPath = remoteHermesHomePath("\(identity.ownershipID)/\(identity.nonce).log")
        let processCheck: String
        if let pid {
            processCheck = "if (tr '\\0' ' ' < /proc/\(pid)/cmdline 2>/dev/null || ps -o command= -p \(pid) 2>/dev/null) | grep -F -- \(shellQuote("hermes serve")) >/dev/null && (tr '\\0' ' ' < /proc/\(pid)/cmdline 2>/dev/null || ps -o command= -p \(pid) 2>/dev/null) | grep -F -- \(shellQuote("--ssh-owner-nonce \(identity.nonce)")) >/dev/null; then kill \(pid) 2>/dev/null || true; fi"
        } else {
            processCheck = "self=$$; parent=$PPID; processList=$(ps -eo pid=,command= 2>/dev/null || true); if [ -n \"$processList\" ]; then printf '%s\\n' \"$processList\" | while read -r candidate commandLine; do [ \"$candidate\" = \"$self\" ] || [ \"$candidate\" = \"$parent\" ] && continue; case \"$commandLine\" in *'hermes serve'*'--ssh-owner-nonce \(identity.nonce)'*) kill \"$candidate\" 2>/dev/null || true;; esac; done; else for candidate in /proc/[0-9]*; do candidate=\"${candidate##*/}\"; [ \"$candidate\" = \"$self\" ] || [ \"$candidate\" = \"$parent\" ] && continue; commandLine=$(tr '\\0' ' ' < /proc/$candidate/cmdline 2>/dev/null || true); case \"$commandLine\" in *'hermes serve'*'--ssh-owner-nonce \(identity.nonce)'*) kill \"$candidate\" 2>/dev/null || true;; esac; done; fi"
        }
        let command = "\(processCheck); rm -f \(tokenPath) \(logPath)"
        _ = try? await sshClient.executeCommand(command)
    }

    func stop() async throws {
        guard listener != nil || client != nil else {
            throw SshTunnelError.notStarted
        }

        var firstError: Error?
        defer {
            listener = nil
            client = nil
            eventLoopGroup = nil
            remoteBackend = nil
        }

        if let listener {
            do {
                try await listener.close()
            } catch {
                firstError = error
            }
        }
        if let remoteBackend {
            await stopRemoteBackend(remoteBackend, using: client)
        }
        if let client {
            do {
                try await client.close()
            } catch {
                firstError = firstError ?? error
            }
        }
        if let eventLoopGroup {
            do {
                try await eventLoopGroup.shutdownGracefully()
            } catch {
                firstError = firstError ?? error
            }
        }

        if let firstError {
            throw firstError
        }
    }
}
