import SwiftUI
import HermesCore
import HermesFeatures

struct ScheduledJobsView: View {
    @Bindable var jobs: ScheduledJobsStore
    let profiles: [JSON]
    let isPreview: Bool
    @State private var editor: JobEditorContext?
    @State private var action: JobActionContext?
    var body: some View {
        List {
            if isPreview { Label("Preview · sample jobs · controls disabled", systemImage: "eye").font(.caption) }
            if let error = jobs.errorMessage { Text(error).foregroundStyle(.red) }
            if let unknown = jobs.unknownOutcomeMessage { Text(unknown).foregroundStyle(.orange) }
            if jobs.loading { ProgressView("Loading scheduled jobs") }
            if jobs.jobs.isEmpty, !jobs.loading {
                ContentUnavailableView("No scheduled jobs", systemImage: "calendar.badge.clock",
                    description: Text(isPreview ? "Sample jobs appear here in preview." : "Refresh to load jobs from your Mac."))
            }
            ForEach(Array(jobs.jobs.enumerated()), id: \.offset) { _, job in
                Section {
                    Button { Task { await jobs.select(job) } } label: {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(job["name"].string ?? "Scheduled job").font(.headline)
                                Spacer()
                                Text(paused(job) ? "Paused" : "Active").font(.caption)
                            }
                            Text(ScheduledJobDraft(job: job).schedule).font(.subheadline).foregroundStyle(.secondary)
                            Text("Profile: \(job["profile"].string ?? "Unknown")").font(.caption).foregroundStyle(.secondary)
                        }.foregroundStyle(.primary)
                    }
                    if let prompt = job["prompt"].string { Text(prompt).font(.callout).lineLimit(3) }
                    if let next = job["next_run_at"].string { Text("Next: \(next)").font(.caption).foregroundStyle(.secondary) }
                    if let error = job["last_error"].string, !error.isEmpty { Text(error).font(.caption).foregroundStyle(.red) }
                    HStack {
                        Button("Edit") { editor = JobEditorContext(job: job, context: jobs.contextID) }
                        Spacer()
                        Menu("Actions") {
                            Button(paused(job) ? "Resume" : "Pause") { setAction(paused(job) ? .resume : .pause, job) }
                            Button("Run now") { setAction(.runNow, job) }
                            Button("Delete", role: .destructive) { setAction(.delete, job) }
                        }
                    }.disabled(isPreview || !jobs.canMutate)
                }
            }
            if let selected = jobs.selectedJob {
                Section("Recent runs · \(selected["name"].string ?? "Job")") {
                    if jobs.runsLoading { ProgressView("Loading history") }
                    else if jobs.runs.isEmpty { Text("No runs recorded").foregroundStyle(.secondary) }
                    ForEach(Array(jobs.runs.enumerated()), id: \.offset) { _, run in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(run["title"].string ?? "Run")
                            Text(run["id"].string ?? run["session_id"].string ?? "Session unavailable")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                    }
                }
            }
        }
        .navigationTitle("Scheduled jobs")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button { Task { await jobs.refresh() } } label: { Image(systemName: "arrow.clockwise") }
                    .accessibilityLabel("Refresh jobs").disabled(isPreview || jobs.loading || jobs.mutating)
                Button { editor = JobEditorContext(job: nil, context: jobs.contextID) } label: { Image(systemName: "plus") }
                    .accessibilityLabel("New scheduled job").disabled(isPreview || !jobs.canMutate)
            }
        }
        .refreshable { await jobs.refresh() }
        .task { if !isPreview { await jobs.refresh() } }
        .sheet(item: $editor) { item in
            JobEditor(jobs: jobs, profiles: profiles, snapshot: item)
        }
        .confirmationDialog(action.map { "\($0.action.title) on your Mac?" } ?? "Confirm action",
                            isPresented: Binding(get: { action != nil }, set: { if !$0 { action = nil } }),
                            titleVisibility: .visible) {
            if let item = action {
                Button(item.action.title, role: item.action == .delete ? .destructive : nil) {
                    action = nil
                    Task { _ = await jobs.perform(item.action, job: item.job, context: item.context) }
                }
            }
            Button("Cancel", role: .cancel) { action = nil }
        } message: {
            Text(action?.action == .runNow
                 ? "Runs the job immediately, including a paused job. A slow response does not mean it stopped."
                 : "This changes the scheduled job on your Mac.")
        }
    }
    private func paused(_ job: JSON) -> Bool { job["state"].string == "paused" || job["enabled"].bool == false }
    private func setAction(_ action: ScheduledJobAction, _ job: JSON) {
        self.action = JobActionContext(action: action, job: job, context: jobs.contextID)
    }
}

private struct JobEditorContext: Identifiable {
    let id = UUID()
    let job: JSON?
    let context: UUID
}
private struct JobActionContext {
    let action: ScheduledJobAction
    let job: JSON
    let context: UUID
}
private struct JobEditor: View {
    @Bindable var jobs: ScheduledJobsStore
    let profiles: [JSON]
    let snapshot: JobEditorContext
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ScheduledJobDraft()
    @State private var profile = ""
    @State private var confirm = false
    @State private var saveError: String?
    private var scriptOnly: Bool {
        snapshot.job?["no_agent"].bool == true ||
        (!(snapshot.job?["script"].string ?? "").isEmpty && (snapshot.job?["prompt"].string ?? "").isEmpty)
    }
    private var valid: Bool {
        !draft.schedule.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !profile.isEmpty && profile != "all" &&
        (scriptOnly || !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) &&
        (snapshot.job == nil || draft.changes(from: snapshot.job!) != .object([:]))
    }
    var body: some View {
        NavigationStack {
            Form {
                Section("Job") {
                    Picker("Profile", selection: $profile) {
                        Text("Choose profile").tag("")
                        ForEach(Array(profiles.enumerated()), id: \.offset) { _, row in
                            Text(row["name"].string ?? "Profile").tag(row["name"].string ?? "")
                        }
                    }.disabled(snapshot.job != nil)
                    TextField("Name", text: $draft.name)
                    TextField("Schedule", text: $draft.schedule).textInputAutocapitalization(.never)
                    TextField("Prompt", text: $draft.prompt, axis: .vertical).lineLimit(4...12)
                    TextField("Model (optional)", text: $draft.model).textInputAutocapitalization(.never)
                    TextField("Delivery target", text: $draft.deliver).textInputAutocapitalization(.never)
                }
                Section {
                    Text("Runs on your Mac. Use a cron expression or a schedule Hermes understands. Delivery defaults to local.")
                    if scriptOnly { Text("Script-only job: unshown script and advanced settings are preserved.") }
                }.font(.footnote).foregroundStyle(.secondary)
                if let error = saveError ?? jobs.errorMessage { Section { Text(error).foregroundStyle(.red) } }
                Button(snapshot.job == nil ? "Create scheduled job" : "Save changed fields") { confirm = true }
                    .disabled(!valid || !jobs.canMutate || snapshot.context != jobs.contextID)
            }
            .navigationTitle(snapshot.job == nil ? "New scheduled job" : "Edit scheduled job")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
            .onAppear { draft = ScheduledJobDraft(job: snapshot.job ?? .null); profile = snapshot.job?["profile"].string ?? "" }
            .confirmationDialog("Save this job on your Mac?", isPresented: $confirm, titleVisibility: .visible) {
                Button("Confirm") {
                    Task {
                        let saved: Bool
                        if let job = snapshot.job {
                            saved = await jobs.update(job, changed: draft.changes(from: job), context: snapshot.context)
                        } else {
                            saved = await jobs.create(draft.body, concreteProfile: profile, context: snapshot.context)
                        }
                        if saved { dismiss() }
                        else { saveError = jobs.unknownOutcomeMessage ?? jobs.errorMessage ?? "Not saved. Connection or profile changed; close and refresh before trying again." }
                    }
                }
            }
        }
    }
}
