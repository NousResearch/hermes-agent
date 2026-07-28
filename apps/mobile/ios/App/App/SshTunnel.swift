import Citadel
import Crypto
import Foundation
import NIO
import NIOPosix
import NIOSSH

struct SshTunnelConfiguration: Sendable {
    let host: String
    let port: Int
    let username: String
    let privateKey: String
    let hostKey: String
    let remoteHost: String
    let remotePort: Int

    init(
        host: String,
        port: Int = 22,
        username: String,
        privateKey: String,
        hostKey: String,
        remoteHost: String = "127.0.0.1",
        remotePort: Int = 9119
    ) {
        self.host = host
        self.port = port
        self.username = username
        self.privateKey = privateKey
        self.hostKey = hostKey
        self.remoteHost = remoteHost
        self.remotePort = remotePort
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

    func setRemoteChannel(_ channel: Channel) {
        remoteChannel = channel
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard let remoteChannel else {
            context.close(promise: nil)
            return
        }
        remoteChannel.writeAndFlush(self.unwrapInboundIn(data), promise: nil)
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

actor SshTunnel {
    private var client: SSHClient?
    private var listener: Channel?
    private var eventLoopGroup: MultiThreadedEventLoopGroup?

    var isRunning: Bool {
        listener?.isActive == true && client?.isConnected == true
    }

    func start(_ configuration: SshTunnelConfiguration) async throws -> UInt16 {
        guard listener == nil else {
            throw SshTunnelError.alreadyStarted
        }

        let privateKey = try Curve25519.Signing.PrivateKey(sshEd25519: configuration.privateKey)
        let hostKey = try NIOSSHPublicKey(openSSHPublicKey: configuration.hostKey)
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        eventLoopGroup = group

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
                                        targetHost: configuration.remoteHost,
                                        targetPort: configuration.remotePort,
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
            return UInt16(localPort)
        } catch {
            try? await group.shutdownGracefully()
            eventLoopGroup = nil
            throw error
        }
    }

    func stop() async throws {
        guard listener != nil || client != nil else {
            throw SshTunnelError.notStarted
        }

        if let listener {
            try await listener.close()
        }
        if let client {
            try await client.close()
        }
        if let eventLoopGroup {
            try await eventLoopGroup.shutdownGracefully()
        }
        listener = nil
        client = nil
        eventLoopGroup = nil
    }
}
