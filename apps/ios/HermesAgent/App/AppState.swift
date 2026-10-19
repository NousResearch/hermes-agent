import SwiftUI
import Combine

/// Top-level app state: connection lifecycle, the shared API/gateway clients,
/// and the currently selected model. Injected into the environment.
@MainActor
final class AppState: ObservableObject {
    enum Phase: Equatable {
        case needsSetup
        case connecting
        case ready
        case failed(String)
    }

    @Published var phase: Phase = .needsSetup
    @Published var connection: Connection
    @Published var availableModels: [ModelInfo] = []
    @Published var selectedModel: String?

    private(set) var api: HermesAPIClient
    private(set) var gateway: TUIGatewayClient

    init() {
        let conn = Connection.load()
        self.connection = conn
        self.api = HermesAPIClient(connection: conn)
        self.gateway = TUIGatewayClient(connection: conn)
        self.phase = conn.isValid ? .connecting : .needsSetup
    }

    /// Validate and persist a new connection, then probe the server.
    func connect(_ newConnection: Connection) async {
        phase = .connecting
        connection = newConnection
        newConnection.save()
        await api.update(connection: newConnection)
        gateway.disconnect()
        gateway = TUIGatewayClient(connection: newConnection)

        do {
            try await api.ping()
            await loadModels()
            phase = .ready
        } catch {
            phase = .failed((error as? HermesError)?.errorDescription ?? error.localizedDescription)
        }
    }

    /// Re-probe an already-saved connection on launch.
    func bootstrap() async {
        guard connection.isValid else { phase = .needsSetup; return }
        await connect(connection)
    }

    func loadModels() async {
        guard let models = try? await api.models() else { return }
        availableModels = models
        if selectedModel == nil { selectedModel = models.first?.id }
    }

    func signOut() {
        Connection.clear()
        gateway.disconnect()
        connection = Connection(baseURL: "", apiKey: "")
        availableModels = []
        selectedModel = nil
        phase = .needsSetup
    }
}
