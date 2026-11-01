import SwiftUI
import HermesFeatures

struct ControlSettings: View {
    @Bindable var store: CompanionStore
    @State private var signOutConfirmation = false
    var body: some View {
        Form {
            Section("Connection") {
                LabeledContent("Mac", value: store.host)
                LabeledContent("State", value: store.isPreview ? "Preview only" : store.isConnected ? "Connected" : "Disconnected")
                Button("Reconnect securely") { Task { await store.reconnect() } }.disabled(store.isPreview || store.connecting)
            }
            Section("New conversations") {
                TextField("Model override (optional)", text: $store.modelOverride).textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("Leave blank to use the agent's configured model. This changes new conversations only; it does not edit your Mac's global settings.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section("Controls on this phone") {
                Label("Chat and stop a running turn", systemImage: "bubble.left.and.bubble.right")
                Label("Create, edit, and move Kanban tasks", systemImage: "rectangle.split.3x1")
                Label("Approve once or deny a request", systemImage: "hand.raised")
                Text("Mac policies still apply. Secret entry, sudo prompts, terminal, files, voice, and global provider settings remain on the Mac in this version.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section {
                if store.isPreview {
                    Button("Exit preview") { store.signOut() }
                }
                Button("Sign out", role: .destructive) { signOutConfirmation = true }.disabled(store.isPreview)
                Text("Signing out disconnects this phone. It does not stop the Mac or any running jobs.").font(.footnote).foregroundStyle(.secondary)
            }
        }
        .confirmationDialog("Sign out of Hermes on this phone?", isPresented: $signOutConfirmation, titleVisibility: .visible) {
            Button("Sign out", role: .destructive) { store.signOut() }
        }
    }
}

struct RequestSheet: View {
    @Bindable var store: CompanionStore
    let request: PendingRequest
    @Environment(\.dismiss) private var dismiss
    @State private var answer = ""
    @State private var confirmApproval = false
    private var current: Bool { store.isConnected && store.session?.needsRefresh == false && store.session?.requests.contains(request) == true }
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    Label(request.kind == .approval ? "Approval requested on Mac" : "Hermes needs your answer", systemImage: "hand.raised")
                        .font(.headline)
                    Text(request.title).font(request.kind == .approval ? .system(.body, design: .monospaced) : .body).textSelection(.enabled)
                    if let reason = request.payload["description"].string ?? request.payload["reason"].string {
                        Text(reason).foregroundStyle(.secondary)
                    }
                    Text("Request: " + request.id).font(.caption.monospaced()).foregroundStyle(.secondary)
                    if !current { Text("This request is stale or disconnected. Refresh on the conversation screen.").foregroundStyle(.orange) }
                    if request.kind == .approval {
                        let choices = request.payload["choices"].array?.compactMap(\.string) ?? ["once", "deny"]
                        Text("Approving lets this specific action run on your Mac. Existing server checks still apply.").font(.callout).foregroundStyle(.secondary)
                        if choices.contains("once") {
                            Button("Approve once") { confirmApproval = true }.buttonStyle(.borderedProminent).disabled(!current || store.controlBusy)
                        }
                        Button("Deny", role: .destructive) { Task { await store.respond(request, answer: "deny"); if store.errorMessage == nil { dismiss() } } }
                            .buttonStyle(.bordered).disabled(!current || store.controlBusy || !choices.contains("deny"))
                    } else if request.payload["questions"].array != nil || request.payload["multi_select"].bool == true {
                        Text("Answer this multi-part question on the Mac. This phone currently supports single-question replies.").foregroundStyle(.secondary)
                    } else {
                        ForEach(request.payload["choices"].array?.compactMap(\.string) ?? [], id: \.self) { choice in
                            Button(choice) { answer = choice }.buttonStyle(.bordered)
                        }
                        TextField("Your answer", text: $answer, axis: .vertical).lineLimit(3...8).textFieldStyle(.roundedBorder)
                        Button("Send answer") { Task { await store.respond(request, answer: answer); if store.errorMessage == nil { dismiss() } } }
                            .buttonStyle(.borderedProminent).disabled(!current || answer.isEmpty || store.controlBusy)
                    }
                    if let error = store.errorMessage { Text(error).font(.callout).foregroundStyle(.red) }
                }.padding(24)
            }
            .navigationTitle(request.kind == .approval ? "Review action" : "Reply")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } } }
            .confirmationDialog("Allow this action on your Mac?", isPresented: $confirmApproval, titleVisibility: .visible) {
                Button("Approve this request once") { Task { await store.respond(request, answer: "once"); if store.errorMessage == nil { dismiss() } } }
            } message: { Text(request.title) }
        }
    }
}
