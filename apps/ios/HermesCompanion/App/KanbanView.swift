import SwiftUI
import HermesCore
import HermesFeatures

private struct TaskSelection: Identifiable {
    let id = UUID()
    let task: JSON?
    let board: String
}
struct KanbanView: View {
    @Bindable var store: CompanionStore
    @State private var selection: TaskSelection?
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Picker("Board", selection: Binding(get: { store.selectedBoard }, set: { slug in Task { await store.selectBoard(slug) } })) {
                    if store.boards.isEmpty { Text("No board loaded").tag("") }
                    ForEach(Array(store.boards.enumerated()), id: \.offset) { _, board in
                        Text(board["name"].string ?? board["slug"].string ?? "Board").tag(board["slug"].string ?? "")
                    }
                }.disabled(store.boardBusy || store.isPreview)
                Spacer()
                Button { Task { await store.refreshBoards() } } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(QuietButton()).accessibilityLabel("Refresh board").disabled(store.boardBusy || !store.isConnected)
                Button { selection = TaskSelection(task: nil, board: store.selectedBoard) } label: { Image(systemName: "plus") }
                    .buttonStyle(QuietButton()).accessibilityLabel("Create task")
                    .disabled(!store.isConnected || !store.controlsCompatible || store.boardNeedsRefresh || store.boardBusy)
            }.padding(.horizontal, 16)
            HStack {
                if store.isPreview { Text("Sample board · not connected") }
                else if store.boardNeedsRefresh { Text("Not current · refresh required") }
                else if let updated = store.boardUpdatedAt { Text("Updated \(updated.formatted(date: .omitted, time: .shortened))") }
                Spacer()
                if store.boardBusy { ProgressView() }
            }.font(.caption).foregroundStyle(.secondary).padding(.horizontal, 20).padding(.bottom, 12)
            if let error = store.errorMessage { Text(error).font(.footnote).foregroundStyle(.red).padding(.horizontal, 20) }
            if store.columns.isEmpty {
                EmptyPanel(title: "Your work, at a glance", detail: "Connect to your Mac and refresh to load Kanban. The plugin must be enabled on Hermes.")
            } else {
                List {
                    ForEach(Array(store.columns.enumerated()), id: \.offset) { _, column in
                        Section {
                            let tasks = column["tasks"].array ?? []
                            if tasks.isEmpty { Text("No tasks").font(.caption).foregroundStyle(.tertiary) }
                            ForEach(Array(tasks.enumerated()), id: \.offset) { _, task in
                                Button { selection = TaskSelection(task: task, board: store.selectedBoard) } label: {
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(task["title"].string ?? "Untitled task").foregroundStyle(.primary).font(.body.weight(.medium))
                                        if let assignee = task["assignee"].string, !assignee.isEmpty {
                                            Label(assignee, systemImage: "person.crop.circle").font(.caption).foregroundStyle(.secondary)
                                        }
                                        if let reason = task["block_reason"].string, !reason.isEmpty { Text(reason).font(.caption).foregroundStyle(.orange).lineLimit(3) }
                                    }.padding(.vertical, 5)
                                }
                            }
                        } header: {
                            HStack { Text((column["name"].string ?? "Tasks").capitalized); Spacer(); Text(String(column["tasks"].array?.count ?? 0)) }
                        }
                    }
                }.listStyle(.plain).refreshable { await store.refreshBoard() }
            }
        }
        .task { if !store.isPreview { await store.refreshBoards() } }
        .sheet(item: $selection) { value in TaskEditor(store: store, task: value.task, board: value.board) }
    }
}

private struct TaskEditor: View {
    @Bindable var store: CompanionStore
    let task: JSON?
    let board: String
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var bodyText = ""
    @State private var assignee = ""
    @State private var status = "triage"
    @State private var confirm = false
    private var editable: Bool { store.isConnected && store.controlsCompatible && !store.boardNeedsRefresh && !store.boardBusy && board == store.selectedBoard }
    var body: some View {
        NavigationStack {
            Form {
                Section("Task") {
                    TextField("Title", text: $title)
                    TextField("Description", text: $bodyText, axis: .vertical).lineLimit(4...12)
                    TextField("Assignee (optional)", text: $assignee).textInputAutocapitalization(.never).autocorrectionDisabled()
                }
                if task != nil {
                    Section("Workflow") {
                        Picker("Status", selection: $status) {
                            ForEach(store.columns.compactMap { $0["name"].string }, id: \.self) { name in Text(name.capitalized).tag(name) }
                        }
                        Text("Status changes are requests to the existing Hermes workflow. Required evidence and transition checks remain in force.").font(.footnote).foregroundStyle(.secondary)
                    }
                } else {
                    Section { Text("New tasks enter triage. Hermes decides when work can be dispatched under the existing workflow.").font(.footnote).foregroundStyle(.secondary) }
                }
                if let reason = task?["block_reason"].string, !reason.isEmpty { Section("Blocker") { Text(reason) } }
                if let result = task?["latest_summary"].string ?? task?["result"].string, !result.isEmpty {
                    Section("Latest result") { Text(result).textSelection(.enabled) }
                }
                Section {
                    Button(task == nil ? "Create task" : "Save changes") { confirm = true }
                        .disabled(!editable || title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if let error = store.errorMessage { Text(error).foregroundStyle(.red).font(.footnote) }
                    if !editable { Text("Connect and refresh this board before changing it.").font(.footnote).foregroundStyle(.secondary) }
                }
            }
            .navigationTitle(task == nil ? "New task" : "Task details").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } } }
            .onAppear {
                title = task?["title"].string ?? ""; bodyText = task?["body"].string ?? ""
                assignee = task?["assignee"].string ?? ""; status = task?["status"].string ?? "triage"
            }
            .confirmationDialog(task == nil ? "Create this task on your Mac?" : "Apply these task changes on your Mac?", isPresented: $confirm, titleVisibility: .visible) {
                Button("Confirm changes") {
                    guard editable else { return }
                    Task {
                        if let task {
                            await store.updateTask(original: task, title: title, body: bodyText, assignee: assignee, status: status)
                        } else { await store.createTask(title: title, body: bodyText, assignee: assignee) }
                        if store.errorMessage == nil { dismiss() }
                    }
                }
            } message: { Text("Board: \(board). Changing a task can affect dispatch and running work.") }
        }
    }
}
