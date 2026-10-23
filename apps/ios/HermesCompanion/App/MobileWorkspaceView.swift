import SwiftUI
import HermesCore
import HermesFeatures

struct MobileWorkspaceView: View {
    @Bindable var store: CompanionStore
    @State private var tab = 0
    var body: some View {
        TabView(selection: $tab) {
            ChatView(store: store).tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }.tag(0)
            NavigationStack { KanbanView(store: store).navigationTitle("Kanban").navigationBarTitleDisplayMode(.inline) }
                .tabItem { Label("Kanban", systemImage: "rectangle.split.3x1") }.tag(1)
            NavigationStack { ScheduledJobsView(jobs: store.scheduledJobs, profiles: store.profiles, isPreview: store.isPreview) }
                .tabItem { Label("Scheduled", systemImage: "calendar.badge.clock") }.tag(2)
            NavigationStack { WorkspaceMenu(store: store) { tab = 0 } }
                .tabItem { Label("Workspace", systemImage: "square.grid.2x2") }.tag(3)
        }
    }
}

struct ProfileSelector: View {
    @Bindable var store: CompanionStore
    var body: some View {
        Picker("Agent profile", selection: Binding(get: { store.selectedProfile }, set: { name in Task { await store.chooseProfile(name) } })) {
            Text("Gateway default").tag("")
            ForEach(Array(store.profiles.enumerated()), id: \.offset) { _, profile in
                Text(profile["display_name"].string.flatMap { $0.isEmpty ? nil : $0 } ?? profile["name"].string ?? "Profile").tag(profile["name"].string ?? "")
            }
        }.disabled(store.isPreview || !store.isConnected || store.controlBusy)
    }
}

private struct WorkspaceMenu: View {
    @Bindable var store: CompanionStore
    let showChat: () -> Void
    @State private var newSession = false
    var body: some View {
        List {
            Section {
                HStack(spacing: 12) {
                    HermesMark(size: 38)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Hermes").font(.headline)
                        Text(store.isPreview ? "Preview · not connected" : store.isConnected ? store.host : "Disconnected · Mac may still be working")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                ProfileSelector(store: store)
                Button { newSession = true } label: { Label("New session", systemImage: "square.and.pencil") }
                    .disabled((!store.isConnected && !store.isPreview) || store.controlBusy)
            }
            Section("Your workspace") {
                NavigationLink { ProfilesView(store: store) } label: { Label("Agent profiles", systemImage: "person.2") }
                NavigationLink { CapabilitiesView(store: store).id(store.selectedProfile) } label: { Label("Capabilities", systemImage: "puzzlepiece.extension") }
                NavigationLink { LibraryView(library: store.library, page: .learning, isPreview: store.isPreview).id(store.selectedProfile) } label: { Label("Memory & Star Map", systemImage: "sparkles") }
                NavigationLink { LibraryView(library: store.library, page: .usage, isPreview: store.isPreview).id(store.selectedProfile) } label: { Label("Usage charts", systemImage: "chart.bar.xaxis") }
            }
            Section {
                NavigationLink { ControlSettings(store: store).navigationTitle("Settings") } label: { Label("Settings", systemImage: "slider.horizontal.3") }
                Text("These views use the same Mac data as Hermes Desktop. Tabs keep your conversation open while you check jobs and tasks.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }.navigationTitle("Workspace")
            .sheet(isPresented: $newSession) { NewSessionView(store: store) { showChat() } }
    }
}

struct NewSessionView: View {
    @Bindable var store: CompanionStore
    var onCreated: () -> Void = {}
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var cwd = ""
    @State private var model = ""
    @State private var provider = ""
    var body: some View {
        NavigationStack {
            Form {
                Section("Session") {
                    ProfileSelector(store: store)
                    TextField("Title (optional)", text: $title)
                    TextField("Mac workspace path (optional)", text: $cwd).textInputAutocapitalization(.never).autocorrectionDisabled()
                }
                Section("Model") {
                    TextField("Profile default", text: Binding(get: { model }, set: { model = $0; provider = "" }))
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    if !provider.isEmpty { LabeledContent("Provider", value: provider) }
                    let providers = store.library.values[.models]?["providers"].array ?? []
                    ForEach(Array(providers.enumerated()), id: \.offset) { _, catalog in
                        let models = catalog["models"].array ?? []
                        if !models.isEmpty {
                            Menu(catalog["name"].string ?? catalog["slug"].string ?? "Models") {
                                ForEach(Array(models.enumerated()), id: \.offset) { _, row in
                                    let id = row.string ?? row["id"].string ?? row["value"].string ?? ""
                                    if !id.isEmpty { Button(row["name"].string ?? id) { model = id; provider = catalog["slug"].string ?? "" } }
                                }
                            }
                        }
                    }
                    Text("Uses this profile's configuration unless you choose a model. Workspace paths refer to your Mac, not your phone.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                if let error = store.errorMessage { Section { Text(error).foregroundStyle(.red) } }
            }.navigationTitle("New session").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button(store.loadingSession ? "Creating…" : "Create session") {
                            store.modelOverride = model
                            Task { await store.createChat(title: title, cwd: cwd, provider: provider); if store.errorMessage == nil, store.session != nil { onCreated(); dismiss() } }
                        }.disabled(!store.isConnected || store.loadingSession || store.controlBusy || !store.controlsCompatible)
                    }
                }
                .task(id: store.selectedProfile) { model = ""; provider = ""; if !store.isPreview { await store.library.refresh(.models) } }
        }
    }
}

private struct ProfilesView: View {
    @Bindable var store: CompanionStore
    var body: some View {
        List {
            ForEach(Array(store.profiles.enumerated()), id: \.offset) { _, profile in
                if let name = profile["name"].string {
                    NavigationLink { ProfileDetail(store: store, name: name) } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(name).font(.headline)
                            Text(profile["description"].string ?? profile["model"].string ?? "Agent profile").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }.navigationTitle("Agent profiles").refreshable { await store.refreshRoster() }
    }
}

private struct ProfileDetail: View {
    @Bindable var store: CompanionStore
    let name: String
    @State private var details: JSON?
    @State private var loading = true
    var body: some View {
        List {
            if loading { ProgressView("Loading profile") }
            else if let details {
                Section("Configuration") {
                    Text(details["description"].string ?? "No description")
                    LabeledContent("Provider", value: details["model"]["provider"].string ?? "Default")
                    LabeledContent("Model", value: details["model"]["default"].string ?? "Default")
                }
                if let soul = details["soul"].string, !soul.isEmpty { Section("SOUL") { Text(soul).textSelection(.enabled) } }
                Section { Button("Use this profile") { Task { await store.chooseProfile(name) } }.disabled(!store.isConnected || store.controlBusy) }
            } else { Text(store.errorMessage ?? "Connect to load this profile.").foregroundStyle(.secondary) }
        }.navigationTitle(name).task { details = await store.describeProfile(name); loading = false }
    }
}
