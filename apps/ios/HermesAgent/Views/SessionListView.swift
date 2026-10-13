import SwiftUI

/// Home screen: the list of Hermes sessions (synced from CLI / other platforms),
/// with a model picker, new-session action, and navigation into chat.
struct SessionListView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var vm: SessionListViewModel
    @State private var showModelPicker = false
    @State private var showSettings = false
    @State private var pushedSession: HermesSession?

    init() {
        // Replaced in `.task` once the environment is available; placeholder keeps
        // the StateObject initializer total.
        _vm = StateObject(wrappedValue: SessionListViewModel(api: HermesAPIClient(connection: Connection.load())))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                content
            }
            .navigationTitle("Sessions")
            .toolbarBackground(Theme.background, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showModelPicker = true } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "cpu")
                            Text(shortModel).lineLimit(1)
                        }
                        .font(.footnote.weight(.medium))
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { Task { await startNewSession() } } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
            .navigationDestination(item: $pushedSession) { session in
                ChatView(session: session)
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView()
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
        }
        .task {
            await vm.refresh()
        }
        .refreshable { await vm.refresh() }
    }

    private var shortModel: String {
        guard let m = appState.selectedModel else { return "Model" }
        return String(m.split(separator: "/").last ?? Substring(m))
    }

    @ViewBuilder
    private var content: some View {
        if vm.isLoading && vm.sessions.isEmpty {
            ProgressView().tint(Theme.accent)
        } else if vm.sessions.isEmpty {
            EmptyStateView(
                icon: "tray",
                title: "No sessions yet",
                message: "Start a conversation or resume one from your CLI. New sessions sync here automatically.",
                actionTitle: "New Session"
            ) { await startNewSession() }
        } else {
            List {
                ForEach(vm.sessions) { session in
                    Button { pushedSession = session } label: {
                        SessionRow(session: session)
                    }
                    .listRowBackground(Theme.background)
                    .listRowSeparatorTint(Theme.stroke)
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            Task { await vm.delete(session) }
                        } label: { Label("Delete", systemImage: "trash") }
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .overlay(alignment: .bottom) {
                if let error = vm.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                        .padding(8)
                }
            }
        }
    }

    private func startNewSession() async {
        if let session = await vm.createSession(model: appState.selectedModel) {
            pushedSession = session
        }
    }
}

private struct SessionRow: View {
    let session: HermesSession

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Theme.surface)
                    .frame(width: 40, height: 40)
                Image(systemName: session.isActive ? "circle.fill" : "checkmark")
                    .font(.system(size: session.isActive ? 8 : 13, weight: .bold))
                    .foregroundStyle(session.isActive ? Theme.success : Theme.textFaint)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(session.displayTitle)
                    .font(.body.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if let model = session.model {
                        Text(String(model.split(separator: "/").last ?? Substring(model)))
                    }
                    if let count = session.messageCount {
                        Text("· \(count) msgs")
                    }
                    if let date = session.lastActiveDate {
                        Text("· \(date.formatted(.relative(presentation: .numeric)))")
                    }
                }
                .font(.caption)
                .foregroundStyle(Theme.textFaint)
                .lineLimit(1)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textFaint)
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }
}
