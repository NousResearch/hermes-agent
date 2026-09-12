import Foundation
import Security

/// Internal seam so transport tests never touch the user's Keychain.
protocol CredentialStoring: AnyObject {
    func save(_ cookies: [HTTPCookie], for origin: URL) throws
    func load(for origin: URL) throws -> [HTTPCookie]
    func clear(for origin: URL) throws
}

final class CredentialStore: CredentialStoring {
    private static let service = "com.hermescompanion.session"

    fileprivate struct StoredCookie: Codable {
        let name: String
        let value: String
        let domain: String
        let path: String
        let expires: Date?
        let secure: Bool
        let httpOnly: Bool
    }

    func save(_ cookies: [HTTPCookie], for origin: URL) throws {
        let account = try account(for: origin)
        let sessionCookies = cookies.compactMap(StoredCookie.init(cookie:))
        let data = try JSONEncoder().encode(sessionCookies)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account,
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item.merge(attributes) { _, new in new }
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else { throw KeychainError(status: addStatus) }
        } else if status != errSecSuccess {
            throw KeychainError(status: status)
        }
    }

    func load(for origin: URL) throws -> [HTTPCookie] {
        let account = try account(for: origin)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return [] }
        guard status == errSecSuccess, let data = result as? Data else { throw KeychainError(status: status) }
        return try JSONDecoder().decode([StoredCookie].self, from: data).compactMap(\.cookie)
    }

    func clear(for origin: URL) throws {
        let account = try account(for: origin)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw KeychainError(status: status) }
    }

    private func account(for origin: URL) throws -> String {
        guard let scheme = origin.scheme?.lowercased(), let host = origin.host?.lowercased() else {
            throw EndpointError.invalidURL
        }
        let port = origin.port.map { ":\($0)" } ?? ""
        return "\(scheme)://\(host)\(port)"
    }
}

private extension CredentialStore.StoredCookie {
    init?(cookie: HTTPCookie) {
        let name = cookie.name.lowercased()
        guard name.hasSuffix("hermes_session_at") || name.hasSuffix("hermes_session_rt") ||
                name.hasSuffix("hermes_session_provider") else { return nil }
        self.init(
            name: cookie.name,
            value: cookie.value,
            domain: cookie.domain,
            path: cookie.path,
            expires: cookie.expiresDate,
            secure: cookie.isSecure,
            httpOnly: cookie.isHTTPOnly
        )
    }

    var cookie: HTTPCookie? {
        var properties: [HTTPCookiePropertyKey: Any] = [
            .name: name,
            .value: value,
            .domain: domain,
            .path: path,
        ]
        if let expires { properties[.expires] = expires }
        if secure { properties[.secure] = "TRUE" }
        if httpOnly { properties[HTTPCookiePropertyKey("HttpOnly")] = "TRUE" }
        return HTTPCookie(properties: properties)
    }
}

private struct KeychainError: LocalizedError {
    let status: OSStatus
    var errorDescription: String? { "Could not securely store the Hermes session (Keychain status \(status))." }
}
