import Capacitor
import Foundation
import Security

@objc(SshTunnelPlugin)
final class SshTunnelPlugin: CAPPlugin, CAPBridgedPlugin {
    let identifier = "SshTunnelPlugin"
    let jsName = "SshTunnel"
    let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "storePrivateKey", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "start", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "stop", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "status", returnType: CAPPluginReturnPromise)
    ]

    private let tunnel = SshTunnel()
    private let keychainService = "com.nousresearch.hermes.mobile.ssh"
    private let sessionKeychainService = "com.nousresearch.hermes.mobile.ssh.session"

    @objc func storePrivateKey(_ call: CAPPluginCall) {
        guard let key = call.getString("privateKey"), !key.isEmpty else {
            call.reject("An Ed25519 private key is required.", "invalid_private_key")
            return
        }

        let account = call.getString("account") ?? "default"
        do {
            try saveKey(key, account: account)
            call.resolve()
        } catch {
            call.reject("Could not store the SSH private key.", "keychain_write_failed", error)
        }
    }

    @objc func start(_ call: CAPPluginCall) {
        guard
            let host = call.getString("host"), !host.isEmpty,
            let username = call.getString("username"), !username.isEmpty,
            let hostKey = call.getString("hostKey"), !hostKey.isEmpty
        else {
            call.reject("SSH host, username and pinned host key are required.", "invalid_configuration")
            return
        }

        let account = call.getString("account") ?? "default"
        guard let privateKey = readKey(account: account) else {
            call.reject("No SSH private key is stored for this account.", "missing_private_key")
            return
        }

        let port = call.getInt("port") ?? 22
        let configuration = SshTunnelConfiguration(
            host: host,
            port: port,
            username: username,
            privateKey: privateKey,
            hostKey: hostKey,
            previousSession: readSession(account: account)
        )

        Task {
            do {
                let localPort = try await tunnel.start(configuration)
                var result: [String: Any] = [
                    "url": "http://127.0.0.1:\(localPort)",
                    "localPort": localPort
                ]
                if let token = await tunnel.sessionToken {
                    result["token"] = token
                }
                if let identity = await tunnel.sessionIdentity {
                    try saveSession(identity, account: account)
                }
                call.resolve(result)
            } catch {
                call.reject(error.localizedDescription, "ssh_tunnel_start_failed", error)
            }
        }
    }

    @objc func stop(_ call: CAPPluginCall) {
        Task {
            do {
                try await tunnel.stop()
                deleteSession(account: call.getString("account") ?? "default")
                call.resolve()
            } catch SshTunnelError.notStarted {
                call.resolve()
            } catch {
                call.reject(error.localizedDescription, "ssh_tunnel_stop_failed", error)
            }
        }
    }

    @objc func status(_ call: CAPPluginCall) {
        Task {
            let running = await tunnel.isRunning
            var result: [String: Any] = ["running": running]
            if let localPort = await tunnel.localPort {
                result["localPort"] = localPort
                result["url"] = "http://127.0.0.1:\(localPort)"
            }
            if let token = await tunnel.sessionToken {
                result["token"] = token
            }
            call.resolve(result)
        }
    }

    private func saveKey(_ key: String, account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: Data(key.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var item = query
            item.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw NSError(domain: NSOSStatusErrorDomain, code: Int(addStatus))
            }
        } else if updateStatus != errSecSuccess {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(updateStatus))
        }
    }

    private func readKey(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    private func sessionQuery(account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: sessionKeychainService,
            kSecAttrAccount as String: account
        ]
    }

    private func saveSession(_ identity: RemoteSessionIdentity, account: String) throws {
        let data = try JSONEncoder().encode(identity)
        let query = sessionQuery(account: account)
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecItemNotFound {
            var item = query
            item.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw NSError(domain: NSOSStatusErrorDomain, code: Int(addStatus))
            }
        } else if updateStatus != errSecSuccess {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(updateStatus))
        }
    }

    private func readSession(account: String) -> RemoteSessionIdentity? {
        var query = sessionQuery(account: account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else {
            return nil
        }
        return try? JSONDecoder().decode(RemoteSessionIdentity.self, from: data)
    }

    private func deleteSession(account: String) {
        SecItemDelete(sessionQuery(account: account) as CFDictionary)
    }
}
