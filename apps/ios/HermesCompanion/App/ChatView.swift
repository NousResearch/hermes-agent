import SwiftUI
import HermesCore
import HermesFeatures

private enum ChatPanel: String, Identifiable { case conversations, kanban, settings; var id: String { rawValue } }
struct ChatView: View {
    @Bindable var store: CompanionStore
    @State private var panel: ChatPanel?
    @State private var draft = ""
    @State private var stopConfirmation = false
    @State private var request: PendingRequest?
    @State private var newSession = false
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                connectionStrip
                if let error = store.errorMessage {
                    HStack(alignment: .top) {
                        Image(systemName: "exclamationmark.circle")
                        Text(error).font(.footnote).frame(maxWidth: .infinity, alignment: .leading)
                        Button { store.errorMessage = nil } label: { Image(systemName: "xmark") }.accessibilityLabel("Dismiss error")
                    }.padding(12).foregroundStyle(.red).background(.red.opacity(0.05))
                }
                if store.loadingSession { Spacer(); ProgressView("Opening conversation"); Spacer() }
                else if let state = store.session {
                    ScrollViewReader { reader in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 25) {
                                ForEach(state.messages) { message in TranscriptRow(message: message) }
                                ForEach(state.requests) { pending in
                                    Button { request = pending } label: {
                                        HStack {
                                            Image(systemName: pending.kind == .approval ? "hand.raised" : "questionmark.bubble")
                                            VStack(alignment: .leading, spacing: 5) {
                                                Text(pending.kind == .approval ? "Review approval" : "Answer Hermes").font(.subheadline.weight(.semibold))
                                                Text(pending.title).font(.caption).lineLimit(3)
                                            }
                                            Spacer(); Image(systemName: "chevron.right")
                                        }.padding(14).background(HermesTheme.surface).clipShape(RoundedRectangle(cornerRadius: 10))
                                    }.buttonStyle(.plain).disabled(!store.isConnected || state.needsRefresh || store.controlBusy)
                                }
                                if state.needsRefresh {
                                    Button("Refresh conversation") { Task { await store.openSession(state.storedID) } }
                                        .buttonStyle(.bordered).disabled(!store.isConnected)
                                }
                                Color.clear.frame(height: 1).id("bottom")
                            }.padding(20)
                        }
                        .onChange(of: state.messages.count) { old, new in
                            if old == 0 || state.delivery == .sending { reader.scrollTo("bottom", anchor: .bottom) }
                        }
                    }
                } else {
                    VStack {
                        EmptyPanel(title: "What shall we work on?", detail: "Pick a conversation from your Mac, or start something new with Hermes.")
                        Button("New session") { newSession = true }
                            .buttonStyle(.borderedProminent).disabled(!store.isConnected).padding(.bottom, 36)
                    }
                }
                composer
            }
            .background(HermesTheme.background)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { panel = .conversations } label: { Image(systemName: "sidebar.left") }
                        .accessibilityLabel("Open conversations")
                }
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 9) { HermesMark(size: 26); Text(store.selectedProfile.isEmpty ? "Hermes" : store.selectedProfile).font(.headline) }
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { newSession = true } label: { Image(systemName: "square.and.pencil") }.accessibilityLabel("New session")
                        .disabled((!store.isConnected && !store.isPreview) || store.controlBusy)
                }
            }
            .sheet(item: $panel) { selection in
                NavigationStack {
                    Group {
                        switch selection {
                        case .conversations: ConversationDrawer(store: store) { panel = nil }
                        case .kanban: KanbanView(store: store)
                        case .settings: ControlSettings(store: store)
                        }
                    }
                    .navigationTitle(selection == .conversations ? "Conversations" : selection == .kanban ? "Kanban" : "Settings")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { panel = nil } } }
                }.presentationDragIndicator(.visible)
            }
            .sheet(item: $request) { pending in RequestSheet(store: store, request: pending) }
            .sheet(isPresented: $newSession) { NewSessionView(store: store) }
            .confirmationDialog("Stop this conversation on your Mac?", isPresented: $stopConfirmation, titleVisibility: .visible) {
                Button("Stop running turn", role: .destructive) { Task { await store.stop() } }
            } message: { Text("This targets only the selected conversation. Completed tool actions cannot be undone.") }
        }
        .onChange(of: store.selectedProfile) { _, _ in draft = ""; request = nil }
        .onChange(of: store.session?.sessionID) { _, _ in draft = ""; request = nil }
    }
    private var connectionStrip: some View {
        HStack(spacing: 7) {
            Circle().fill(store.isConnected ? .green : .orange).frame(width: 6, height: 6)
            Text(store.isPreview ? "Preview · not connected" : store.isConnected ? "Connected" : "Disconnected")
            Spacer()
            if !store.isConnected && !store.isPreview {
                Button(store.connecting ? "Connecting" : "Reconnect") { Task { await store.reconnect() } }.disabled(store.connecting)
            } else { Text(store.isPreview ? "Sample content" : "Mac · Tailscale") }
        }.font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 20).padding(.vertical, 9)
            .background(HermesTheme.surface.opacity(0.6))
    }
    private var composer: some View {
        VStack(alignment: .leading, spacing: 10) {
            Rectangle().fill(HermesTheme.hairline).frame(height: 1)
            HStack(spacing: 8) {
                Image(systemName: "circle.hexagongrid").foregroundStyle(HermesTheme.accent)
                Text(store.currentModel.isEmpty ? "Profile default" : store.currentModel).lineLimit(1)
                Spacer()
                Text(store.session?.activity ?? "Ready when you are").lineLimit(1)
            }.font(.caption2).foregroundStyle(.secondary)
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Message Hermes…", text: $draft, axis: .vertical)
                    .lineLimit(1...6).padding(13).background(HermesTheme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 10)).accessibilityLabel("Message Hermes")
                if store.session?.running == true {
                    Button { stopConfirmation = true } label: {
                        Image(systemName: "stop.fill").frame(width: 44, height: 44).background(HermesTheme.surface).clipShape(Circle())
                    }.accessibilityLabel("Stop conversation").disabled(!store.isConnected || store.controlBusy)
                } else {
                    Button {
                        let message = draft; draft = ""
                        Task { await store.send(message) }
                    } label: {
                        Image(systemName: "arrow.up").fontWeight(.semibold).foregroundStyle(.white)
                            .frame(width: 44, height: 44).background(HermesTheme.accent).clipShape(Circle())
                    }.accessibilityLabel("Send message")
                        .disabled(!store.isConnected || store.session?.canSend != true || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }.padding(.horizontal, 18).padding(.bottom, 12)
    }
}

private struct TranscriptRow: View {
    let message: ChatMessage
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                if message.role == "assistant" { HermesMark(size: 22) }
                else { Image(systemName: message.role == "user" ? "person.crop.circle" : "terminal").foregroundStyle(.secondary) }
                Text(message.role == "assistant" ? "Hermes" : message.role == "user" ? "You" : message.role.capitalized)
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
            Text(.init(message.text)).font(message.role == "tool" ? .system(.callout, design: .monospaced) : .body)
                .textSelection(.enabled).lineSpacing(5).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct ConversationDrawer: View {
    @Bindable var store: CompanionStore
    let close: () -> Void
    @State private var query = ""
    @State private var newSession = false
    var body: some View {
        List {
            Section("Agents") {
                Picker("Profile", selection: Binding(get: { store.selectedProfile }, set: { value in Task { await store.chooseProfile(value) } })) {
                    Text("Gateway default").tag("")
                    ForEach(Array(store.profiles.enumerated()), id: \.offset) { _, row in Text(row["display_name"].string.flatMap { $0.isEmpty ? nil : $0 } ?? row["name"].string ?? "Agent").tag(row["name"].string ?? "") }
                }.disabled(store.controlBusy || store.isPreview)
            }
            Section {
                Button { newSession = true } label: { Label("New session", systemImage: "square.and.pencil") }
                    .disabled(!store.isConnected || store.controlBusy)
            }
            Section("Recent") {
                if store.sessions.isEmpty { Text("No conversations loaded").foregroundStyle(.secondary) }
                ForEach(Array(store.sessions.filter { query.isEmpty || (($0["title"].string ?? "") + " " + ($0["preview"].string ?? "")).localizedCaseInsensitiveContains(query) }.enumerated()), id: \.offset) { _, row in
                    Button {
                        guard let id = row["id"].string else { return }
                        close(); Task { await store.openSession(id) }
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(row["title"].string.flatMap { $0.isEmpty ? nil : $0 } ?? "Untitled conversation").foregroundStyle(.primary)
                            Text(row["preview"].string ?? "").lineLimit(2).font(.caption).foregroundStyle(.secondary)
                        }.padding(.vertical, 5)
                    }.disabled(!store.isConnected || store.controlBusy)
                }
            }
        }.listStyle(.plain).searchable(text: $query, prompt: "Search loaded sessions")
            .refreshable { await store.refreshRoster() }
            .sheet(isPresented: $newSession) { NewSessionView(store: store) { close() } }
    }
}
