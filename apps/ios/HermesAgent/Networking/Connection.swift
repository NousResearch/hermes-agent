import Foundation

/// Self-hosted connection settings. The base URL lives in UserDefaults (not
/// secret); the API key is kept in the Keychain.
struct Connection: Equatable {
    var baseURL: String
    var apiKey: String

    static let baseURLKey = "hermes.baseURL"
    static let apiKeyKeychain = "hermes.apiKey"

    var url: URL? { URL(string: baseURL) }

    var isValid: Bool {
        guard let u = url, let scheme = u.scheme else { return false }
        return (scheme == "http" || scheme == "https") && u.host != nil
    }

    /// WebSocket URL for the TUI gateway (`/api/ws`), derived from the base URL.
    var webSocketURL: URL? {
        guard var components = url.flatMap({ URLComponents(url: $0, resolvingAgainstBaseURL: false) }) else { return nil }
        components.scheme = (components.scheme == "https") ? "wss" : "ws"
        // When a host is present, URLComponents requires the path to begin with
        // "/" (otherwise `.url` returns nil). Normalize the join accordingly.
        let base = components.path.hasSuffix("/") ? String(components.path.dropLast()) : components.path
        components.path = base + "/api/ws"
        return components.url
    }

    static func load() -> Connection {
        let base = UserDefaults.standard.string(forKey: baseURLKey) ?? ""
        let key = KeychainStore.get(apiKeyKeychain) ?? ""
        return Connection(baseURL: base, apiKey: key)
    }

    func save() {
        UserDefaults.standard.set(baseURL, forKey: Self.baseURLKey)
        if apiKey.isEmpty {
            KeychainStore.delete(Self.apiKeyKeychain)
        } else {
            KeychainStore.set(apiKey, for: Self.apiKeyKeychain)
        }
    }

    static func clear() {
        UserDefaults.standard.removeObject(forKey: baseURLKey)
        KeychainStore.delete(apiKeyKeychain)
    }
}

enum HermesError: LocalizedError {
    case notConfigured
    case badURL
    case unauthorized
    case server(Int, String)
    case decoding(String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured: return "No server configured. Add your server URL and API key."
        case .badURL: return "The server URL is invalid."
        case .unauthorized: return "Authentication failed. Check your API key."
        case .server(let code, let msg): return "Server error (\(code)): \(msg)"
        case .decoding(let m): return "Couldn't read the server response: \(m)"
        case .transport(let m): return m
        }
    }
}
