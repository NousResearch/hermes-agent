"""Pre-flight skill check: a task whose forced skill cannot load on the assignee's
profile is healed from the fleet source or parked needs_input — never spawned into
a crash loop (worker-deaths.log reason=unknown_skill)."""

from types import SimpleNamespace

from hermes_cli import kanban_worker_readiness as preflight


def _task(skills, assignee="reviewer"):
    return SimpleNamespace(skills=skills, assignee=assignee, id="t1", current_run_id=7)


class TestPreflightWorkerSkills:
    def test_no_skills_or_home_skips_check(self):
        assert preflight.preflight_worker_skills(_task([]), "/home") is None
        assert preflight.preflight_worker_skills(_task(["sdlc-review"]), None) is None
        assert preflight.preflight_worker_skills(_task(None), "/home") is None

    def test_resolving_skills_proceed(self, monkeypatch):
        monkeypatch.setattr(
            preflight, "_preflight_resolve_skills", lambda home, skills: (list(skills), []))
        assert preflight.preflight_worker_skills(_task(["sdlc-review"]), "/home") is None

    def test_fleet_install_heals_and_proceeds(self, monkeypatch):
        resolutions = iter([(["sdlc-review"], ["sdlc-review"]), (["sdlc-review"], [])])
        monkeypatch.setattr(
            preflight, "_preflight_resolve_skills", lambda home, skills: next(resolutions))
        installed = []
        monkeypatch.setattr(
            preflight, "_install_skill_from_fleet",
            lambda home, name: installed.append(name) or "/fleet/skills/sdlc-review")
        assert preflight.preflight_worker_skills(_task(["sdlc-review"]), "/home") is None
        assert installed == ["sdlc-review"]

    def test_unhealable_skill_parks_with_profile_and_reason(self, monkeypatch):
        monkeypatch.setattr(
            preflight, "_preflight_resolve_skills", lambda home, skills: ([], list(skills)))
        monkeypatch.setattr(preflight, "_install_skill_from_fleet", lambda home, name: None)
        monkeypatch.setattr(preflight, "_fleet_skill_roots", lambda home: [])
        message = preflight.preflight_worker_skills(_task(["sdlc-review"]), "/home")
        assert message is not None
        assert "reviewer" in message and "sdlc-review" in message
        assert "not started" in message

    def test_profile_home_unresolvable(self):
        assert preflight._profile_home_for("") is None
        assert preflight._profile_home_for("no-such-profile-xyz") is None


class TestFindFleetSkill:
    def _write_skill(self, root, rel, name=None):
        skill_dir = root / rel
        skill_dir.mkdir(parents=True, exist_ok=True)
        front = f"name: {name}\n" if name else ""
        (skill_dir / "SKILL.md").write_text(
            f"---\ndescription: d.\n{front}---\n\n# x\n", encoding="utf-8")
        return skill_dir

    def test_category_path_bare_dir_and_frontmatter_name(self, tmp_path):
        fleet = tmp_path / "fleet" / "skills"
        self._write_skill(fleet, "devops/sdlc-review")
        self._write_skill(fleet, "lonely")
        self._write_skill(fleet, "nested/deep", name="front-named")
        root, found = preflight._find_fleet_skill("devops/sdlc-review", [fleet])
        assert root == fleet and found.name == "sdlc-review"
        _root, found = preflight._find_fleet_skill("lonely", [fleet])
        assert found.name == "lonely"
        _root, found = preflight._find_fleet_skill("front-named", [fleet])
        assert found.name == "deep"

    def test_miss_returns_none(self, tmp_path):
        fleet = tmp_path / "fleet" / "skills"
        fleet.mkdir(parents=True)
        assert preflight._find_fleet_skill("absent", [fleet]) is None
        assert preflight._find_fleet_skill("", [fleet]) is None
