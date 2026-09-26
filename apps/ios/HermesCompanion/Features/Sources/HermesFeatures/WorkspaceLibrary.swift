import Foundation
import Observation
import HermesCore

public enum LibraryPage: String, CaseIterable, Hashable, Identifiable {
    case skills, tools, learning, usage, models
    public var id: String { rawValue }
    public var title: String {
        switch self { case .skills: "Skills"; case .tools: "Tools"; case .learning: "Memory & Star Map"; case .usage: "Usage"; case .models: "Models" }
    }
    var path: String {
        switch self { case .skills: "/api/skills"; case .tools: "/api/tools/toolsets"; case .learning: "/api/learning/graph"; case .usage: "/api/analytics/usage"; case .models: "/api/model/options" }
    }
    func accepts(_ value: JSON) -> Bool {
        switch self {
        case .skills, .tools: value.array != nil
        case .learning: value["nodes"].array != nil && value["memory"].array != nil
        case .usage: value["daily"].array != nil && value["totals"].object != nil
        case .models: value["providers"].array != nil
        }
    }
}

/// Desktop catalogs use their real per-endpoint profile query contracts.
@MainActor @Observable public final class WorkspaceLibrary {
    public private(set) var values: [LibraryPage: JSON] = [:]
    public private(set) var fresh: Set<LibraryPage> = []
    public private(set) var loading: Set<LibraryPage> = []
    public private(set) var errors: [LibraryPage: String] = [:]
    public private(set) var profile = ""
    public private(set) var connected = false
    public private(set) var mutating = false
    private var generation = UUID()
    private let transport: CompanionTransport

    public init(transport: CompanionTransport) { self.transport = transport }
    public func configure(connected: Bool, profile: String) {
        guard connected != self.connected || profile != self.profile else { return }
        self.connected = connected; self.profile = profile; generation = UUID()
        values = [:]; fresh = []; loading = []; errors = [:]; mutating = false
    }
    private func query(_ extra: [String: String] = [:]) -> [String: String] {
        var result = extra
        if !profile.isEmpty { result["profile"] = profile }
        return result
    }
    public func refresh(_ page: LibraryPage) async {
        guard connected, !loading.contains(page), !mutating else { return }
        let token = generation
        loading.insert(page); fresh.remove(page); errors[page] = nil
        do {
            let extra = page == .usage ? ["days": "30"] : page == .models ? ["explicit_only": "1"] : [:]
            let value = try await transport.http(page.path, method: "GET", body: nil, query: query(extra))
            guard token == generation else { return }
            guard page.accepts(value) else { throw FeatureError.invalid("Unexpected \(page.title) response. Update the companion before using this view.") }
            values[page] = value; fresh.insert(page); loading.remove(page)
        } catch {
            guard token == generation else { return }
            loading.remove(page); errors[page] = error.localizedDescription
        }
    }
    public func detail(_ page: LibraryPage, id: String) async -> JSON? {
        guard connected, fresh.contains(page), !id.isEmpty else { return nil }
        let token = generation
        do {
            let path: String; let extra: [String: String]
            if page == .skills { path = "/api/skills/content"; extra = ["name": id] }
            else if page == .learning { path = "/api/learning/node"; extra = ["id": id] }
            else { return nil }
            let result = try await transport.http(path, method: "GET", body: nil, query: query(extra))
            guard token == generation else { return nil }
            guard result["content"].string != nil, page != .learning || result["ok"].bool == true else {
                throw FeatureError.invalid("Content is unavailable. Refresh the list.")
            }
            return result
        } catch { if token == generation { errors[page] = error.localizedDescription }; return nil }
    }
    public func setEnabled(_ page: LibraryPage, name: String, enabled: Bool, expectedProfile: String) async -> Bool {
        guard connected, profile == expectedProfile, fresh.contains(page), !mutating,
              page == .skills || page == .tools,
              values[page]?.array?.contains(where: { $0["name"].string == name && $0["enabled"].bool != nil }) == true else { return false }
        let token = generation
        mutating = true; errors[page] = nil
        do {
            // Toolset names are path segments; skill names are JSON fields.
            let path = page == .skills ? "/api/skills/toggle" : "/api/tools/toolsets/" + (try Self.pathID(name))
            let body: JSON = .object(page == .skills ? ["name": .string(name), "enabled": .bool(enabled)] : ["enabled": .bool(enabled)])
            let receipt = try await transport.http(path, method: "PUT", body: body, query: query())
            guard token == generation else { return false }
            guard receipt["ok"].bool == true, receipt["name"].string == name, receipt["enabled"].bool == enabled else {
                throw FeatureError.invalid("The Mac did not confirm this capability change.")
            }
            mutating = false; await refresh(page)
            return fresh.contains(page)
        } catch {
            guard token == generation else { return false }
            mutating = false; fresh.remove(page)
            errors[page] = "Change not confirmed. Refresh before retrying. " + error.localizedDescription
            return false
        }
    }
    public static func pathID(_ id: String) throws -> String {
        guard !id.isEmpty, id != ".", id != "..",
              let result = id.addingPercentEncoding(withAllowedCharacters: .alphanumerics.union(CharacterSet(charactersIn: "_-"))) else {
            throw FeatureError.invalid("Missing or invalid item identity.")
        }
        return result
    }

    #if DEBUG
    public func installPreview(_ page: LibraryPage, value: JSON) {
        guard !connected else { return }
        values[page] = value
    }
    #endif
}
