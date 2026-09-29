import SwiftUI

@MainActor
final class SessionListViewModel: ObservableObject {
    @Published var sessions: [HermesSession] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: HermesAPIClient

    init(api: HermesAPIClient) {
        self.api = api
    }

    func refresh() async {
        isLoading = true
        errorMessage = nil
        do {
            sessions = try await api.sessions(limit: 100)
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
        }
        isLoading = false
    }

    func createSession(model: String?) async -> HermesSession? {
        do {
            let session = try await api.createSession(model: model)
            sessions.insert(session, at: 0)
            return session
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
            return nil
        }
    }

    func delete(_ session: HermesSession) async {
        do {
            try await api.deleteSession(id: session.id)
            sessions.removeAll { $0.id == session.id }
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
        }
    }

    func rename(_ session: HermesSession, to title: String) async {
        do {
            try await api.renameSession(id: session.id, title: title)
            if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
                sessions[idx].title = title
            }
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
        }
    }
}
