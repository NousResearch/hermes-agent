import Foundation

public enum EndpointError: LocalizedError, Sendable {
    case invalidURL
    case requiresHTTPS
    case requiresTailnetOrigin
    case disallowedURLComponent
    case disallowedPath

    public var errorDescription: String? {
        switch self {
        case .invalidURL: return "Enter a complete HTTPS Tailscale hostname."
        case .requiresHTTPS: return "Hermes endpoints must use HTTPS."
        case .requiresTailnetOrigin: return "Hermes endpoints must use a .ts.net hostname."
        case .disallowedURLComponent: return "Endpoint URLs cannot include credentials, a query, or a fragment."
        case .disallowedPath: return "Endpoint URLs must be an origin without a path."
        }
    }
}

/// A validated, root HTTPS origin for a Hermes service published through Tailscale Serve.
public struct Endpoint: Hashable, Sendable {
    public let baseURL: URL

    public init(_ raw: String) throws {
        guard !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              var components = URLComponents(string: raw),
              let scheme = components.scheme?.lowercased(),
              let host = components.host?.lowercased() else {
            throw EndpointError.invalidURL
        }
        guard scheme == "https" else { throw EndpointError.requiresHTTPS }
        guard host.hasSuffix(".ts.net"), host.count > ".ts.net".count else {
            throw EndpointError.requiresTailnetOrigin
        }
        guard components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else {
            throw EndpointError.disallowedURLComponent
        }
        let encodedPath = components.percentEncodedPath.lowercased()
        guard components.path.isEmpty || components.path == "/",
              !encodedPath.contains(".."), !encodedPath.contains("%2e") else {
            throw EndpointError.disallowedPath
        }

        components.scheme = "https"
        components.host = host
        components.path = ""
        components.percentEncodedPath = ""
        guard let url = components.url else { throw EndpointError.invalidURL }
        self.baseURL = url
    }
}
