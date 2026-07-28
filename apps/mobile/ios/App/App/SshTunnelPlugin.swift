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
        let remotePort = call.getInt("remotePort") ?? 9119
        let configuration = SshTunnelConfiguration(
            host: host,
            port: port,
            username: username,
            privateKey: privateKey,
            hostKey: hostKey,
            remotePort: remotePort
        )

        Task {
            do {
                let localPort = try await tunnel.start(configuration)
                call.resolve([
                    "url": "http://127.0.0.1:\(localPort)",
                    "localPort": localPort
                ])
            } catch {
                call.reject(error.localizedDescription, "ssh_tunnel_start_failed", error)
            }
        }
    }

    @objc func stop(_ call: CAPPluginCall) {
        Task {
            do {
                try await tunnel.stop()
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
            call.resolve(["running": await tunnel.isRunning])
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
}
