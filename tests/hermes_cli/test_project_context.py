from hermes_cli import project_context as pc


class FakeSessionDB:
    def __init__(self):
        self.meta = {}

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value


def test_project_context_round_trips_through_state_meta(monkeypatch):
    db = FakeSessionDB()
    monkeypatch.setattr(pc, "_get_session_db", lambda: db)

    ctx = pc.ActiveProjectContext(
        project_id="pci-010",
        project_name="AI optimization",
        capsule_path="/capsules/010",
        metadata={"track": 10, "empty": "", "nested": {"skip": True}},
    )

    assert pc.save_project_context("session-1", ctx) is None
    loaded = pc.load_project_context("session-1")

    assert loaded is not None
    assert loaded.project_id == "pci-010"
    assert loaded.project_name == "AI optimization"
    assert loaded.capsule_path == "/capsules/010"
    prompt = "\n".join(loaded.prompt_lines())
    assert "Current Project Context" in prompt
    assert "track=10" in prompt
    assert "nested" not in prompt


def test_durable_write_intent_requires_scope_and_source_for_active_project():
    active = pc.ActiveProjectContext(project_id="pci-010", project_name="AI optimization")

    rejection = pc.validate_durable_write_intent(
        intent=pc.DurableWriteIntent(tool_name="memory", action="add", destination="memory"),
        active_project=active,
    )

    assert rejection is not None
    assert "Provide an explicit `scope`" in rejection


def test_project_derived_global_write_requires_explicit_approval():
    active = pc.ActiveProjectContext(project_id="pci-010", project_name="AI optimization")

    rejection = pc.validate_durable_write_intent(
        intent=pc.DurableWriteIntent(
            tool_name="skill_manage",
            action="create",
            destination="workflow",
            scope="project",
            source_reference="/capsules/010/source.md",
            project_id="pci-010",
        ),
        active_project=active,
    )

    assert rejection is not None
    assert "without `approved_global=true`" in rejection

    allowed = pc.validate_durable_write_intent(
        intent=pc.DurableWriteIntent(
            tool_name="skill_manage",
            action="create",
            destination="workflow",
            scope="project",
            source_reference="/capsules/010/source.md",
            project_id="pci-010",
            approved_global=True,
        ),
        active_project=active,
    )
    assert allowed is None


def test_clear_project_context_stores_empty_marker(monkeypatch):
    db = FakeSessionDB()
    monkeypatch.setattr(pc, "_get_session_db", lambda: db)
    assert pc.save_project_context("session-1", pc.ActiveProjectContext("p", "Project")) is None

    assert pc.clear_project_context("session-1") is True
    assert db.meta["project:session-1"] == ""
    assert pc.load_project_context("session-1") is None
