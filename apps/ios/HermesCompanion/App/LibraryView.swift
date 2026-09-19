import SwiftUI
import Charts
import HermesCore
import HermesFeatures

struct CapabilitiesView: View {
    @Bindable var store: CompanionStore
    @State private var page: LibraryPage = .skills
    var body: some View {
        VStack(spacing: 0) {
            Picker("Capabilities", selection: $page) {
                Text("Skills").tag(LibraryPage.skills)
                Text("Tools").tag(LibraryPage.tools)
            }.pickerStyle(.segmented).padding()
            LibraryView(library: store.library, page: page, isPreview: store.isPreview).id(page)
        }.navigationTitle("Capabilities")
    }
}

struct LibraryView: View {
    @Bindable var library: WorkspaceLibrary
    let page: LibraryPage
    let isPreview: Bool
    @State private var search = ""
    var body: some View {
        List {
            Section {
                Text(isPreview ? "Preview · not connected" : library.profile.isEmpty ? "Gateway default profile" : "Profile: \(library.profile)")
                    .font(.caption).foregroundStyle(.secondary)
                if library.loading.contains(page) { ProgressView("Loading \(page.title.lowercased())") }
                if let error = library.errors[page] { Text(error).font(.callout).foregroundStyle(.red) }
                if !isPreview, !library.fresh.contains(page), library.values[page] != nil { Text("Not current · refresh required").font(.caption).foregroundStyle(.orange) }
            }
            if let value = library.values[page] {
                if page == .usage { UsageSections(value: value) }
                else if page == .learning { memory(value) }
                else { capabilities(value) }
            } else if !library.loading.contains(page) {
                Text(isPreview ? "This view loads your Mac's real \(page.title.lowercased()) after connection." : "Connect and refresh to load \(page.title.lowercased()).")
                    .foregroundStyle(.secondary)
            }
        }.navigationTitle(page.title).navigationBarTitleDisplayMode(.inline)
            .searchable(text: $search, prompt: "Filter \(page.title.lowercased())")
            .refreshable { await library.refresh(page) }
            .toolbar { Button { Task { await library.refresh(page) } } label: { Image(systemName: "arrow.clockwise") }
                .accessibilityLabel("Refresh \(page.title)").disabled(!library.connected || library.loading.contains(page)) }
            .task { if !isPreview { await library.refresh(page) } }
    }
    @ViewBuilder private func capabilities(_ value: JSON) -> some View {
        let rows = (value.array ?? []).filter { search.isEmpty || (($0["name"].string ?? "") + " " + ($0["description"].string ?? "")).localizedCaseInsensitiveContains(search) }
        Section {
            if rows.isEmpty { Text("No matching \(page.title.lowercased())").foregroundStyle(.secondary) }
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                NavigationLink { CapabilityDetail(library: library, page: page, row: row, profile: library.profile) } label: {
                    VStack(alignment: .leading, spacing: 7) {
                        HStack {
                            Text(row["label"].string ?? row["name"].string ?? "Capability").font(.headline)
                            Spacer()
                            if let enabled = row["enabled"].bool { Text(enabled ? "Enabled" : "Disabled").font(.caption).foregroundStyle(enabled ? Color.green : Color.secondary) }
                        }
                        Text(row["description"].string ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(3)
                        if row["configured"].bool == false { Text("Setup required on Mac").font(.caption).foregroundStyle(.orange) }
                    }.padding(.vertical, 4)
                }
            }
        }
    }
    @ViewBuilder private func memory(_ value: JSON) -> some View {
        let nodes = (value["nodes"].array ?? []).filter { search.isEmpty || ($0["label"].string ?? "").localizedCaseInsensitiveContains(search) }
        Section("Star Map") {
            LabeledContent("Nodes", value: String(value["nodes"].array?.count ?? 0))
            LabeledContent("Connections", value: String(value["edges"].array?.count ?? 0))
            Text("The same memories and learned skills, arranged as a phone-friendly list.").font(.footnote).foregroundStyle(.secondary)
            ForEach(Array(nodes.enumerated()), id: \.offset) { _, row in
                if let id = row["id"].string {
                    NavigationLink { LibraryContent(library: library, page: .learning, id: id, title: row["label"].string ?? "Memory") } label: {
                        Label(row["label"].string ?? id, systemImage: row["kind"].string == "skill" ? "puzzlepiece.extension" : "brain")
                    }
                }
            }
        }
        Section("Memory cards") {
            ForEach(Array((value["memory"].array ?? []).enumerated()), id: \.offset) { _, row in
                DisclosureGroup(row["title"].string ?? "Memory") { Text(row["body"].string ?? "").textSelection(.enabled) }
            }
        }
    }
}

private struct CapabilityDetail: View {
    @Bindable var library: WorkspaceLibrary
    let page: LibraryPage
    let row: JSON
    let profile: String
    @State private var confirm = false
    private var name: String { row["name"].string ?? "" }
    private var current: JSON { library.values[page]?.array?.first { $0["name"].string == name } ?? row }
    var body: some View {
        List {
            Section {
                Text(row["description"].string ?? "")
                LabeledContent("Profile", value: profile.isEmpty ? "Gateway default" : profile)
                if let category = row["category"].string { LabeledContent("Category", value: category) }
                if let enabled = current["enabled"].bool {
                    Button(enabled ? "Disable on Mac" : "Enable on Mac") { confirm = true }
                        .disabled(!library.connected || library.profile != profile || !library.fresh.contains(page) || library.mutating)
                }
                if let error = library.errors[page] { Text(error).foregroundStyle(.red) }
            }
            if page == .skills {
                NavigationLink("Read skill instructions") { LibraryContent(library: library, page: .skills, id: name, title: name) }
            } else {
                Section("Tools") { ForEach(current["tools"].array?.compactMap(\.string) ?? [], id: \.self) { Text($0).font(.callout.monospaced()) } }
            }
        }.navigationTitle(row["label"].string ?? name)
            .confirmationDialog("Change \(name) on your Mac?", isPresented: $confirm, titleVisibility: .visible) {
                Button(current["enabled"].bool == true ? "Disable capability" : "Enable capability") {
                    let enable = current["enabled"].bool != true
                    Task { _ = await library.setEnabled(page, name: name, enabled: enable, expectedProfile: profile) }
                }
            } message: {
                Text(page == .tools ? "This changes the selected profile's tools. Enabling can install or start companion setup on the Mac; credentials may still need Mac setup." : "This changes skill availability for the selected profile. Existing Mac policies remain in force.")
            }
    }
}

private struct LibraryContent: View {
    @Bindable var library: WorkspaceLibrary
    let page: LibraryPage
    let id: String
    let title: String
    @State private var content: String?
    @State private var loading = true
    var body: some View {
        ScrollView {
            if loading { ProgressView("Loading content").padding() }
            else { Text(content ?? library.errors[page] ?? "Content unavailable. Refresh the list.").textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(20) }
        }.navigationTitle(title).navigationBarTitleDisplayMode(.inline)
            .task { content = await library.detail(page, id: id)?["content"].string; loading = false }
    }
}

private struct UsageSections: View {
    let value: JSON
    private func number(_ json: JSON) -> Double? { if case .number(let number) = json { return number }; return nil }
    var body: some View {
        Section("Last 30 days") {
            LabeledContent("Sessions", value: value["totals"]["total_sessions"].int.map(String.init) ?? "—")
            LabeledContent("API calls", value: value["totals"]["total_api_calls"].int.map(String.init) ?? "—")
            LabeledContent("Input tokens", value: value["totals"]["total_input"].int.map(String.init) ?? "—")
            LabeledContent("Output tokens", value: value["totals"]["total_output"].int.map(String.init) ?? "—")
            if let cost = number(value["totals"]["total_estimated_cost"]) { LabeledContent("Estimated cost", value: cost.formatted(.currency(code: "USD"))) }
        }
        Section("Sessions per day") {
            let rows = value["daily"].array ?? []
            if rows.isEmpty { Text("No usage recorded").foregroundStyle(.secondary) }
            else {
                Chart(Array(rows.enumerated()), id: \.offset) { _, row in
                    if let day = row["day"].string, let sessions = row["sessions"].int {
                        BarMark(x: .value("Day", day), y: .value("Sessions", sessions)).foregroundStyle(HermesTheme.accent)
                    }
                }.frame(height: 200).chartXAxis(.hidden).accessibilityLabel("Sessions per day over the last 30 days")
                ForEach(Array(rows.suffix(7).enumerated()), id: \.offset) { _, row in
                    LabeledContent(row["day"].string ?? "Day", value: row["sessions"].int.map(String.init) ?? "—")
                }
            }
        }
        Section("Models") {
            ForEach(Array((value["by_model"].array ?? []).enumerated()), id: \.offset) { _, row in
                LabeledContent(row["model"].string ?? "Model", value: "\(row["sessions"].int ?? 0) sessions")
            }
        }
    }
}
