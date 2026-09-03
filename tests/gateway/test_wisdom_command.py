"""Contract tests for the presentation-neutral `/wisdom` gateway controller."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import gateway.wisdom_command as command_module
from gateway.wisdom_command import (
    WisdomCommandContext,
    WisdomCommandController,
    _CallbackTokens,
    bind_view_callbacks,
    command_error_text,
    issue_continuation,
    render_local_view,
    resolve_continuation,
)


def _context(**overrides) -> WisdomCommandContext:
    values = {
        "user_id": "user-1",
        "chat_id": "chat-1",
        "profile": "default",
        "organization_id": "org-1",
        "is_group": False,
    }
    values.update(overrides)
    return WisdomCommandContext(**values)


class _Service:
    def __init__(self) -> None:
        self.store = SimpleNamespace(active_org_id=lambda: "org-1")
        self.calls: list[tuple] = []

    def status(self):
        return {
            "configured": True,
            "gateway_available": True,
            "error_kind": None,
            "capability_advertised": True,
            "entitled": True,
            "dogfood_admin_claim": True,
            "verified_org_id": "org-1",
            "installation_id": "installation-1",
            "display_scopes": ["wisdom:read", "wisdom:install"],
            "pending_operations": [],
        }

    def setup(self, *, disclosure_accepted=False):
        self.calls.append(("setup", disclosure_accepted))
        return {"organization_id": "org-1"}

    def command_home(self):
        self.calls.append(("command_home",))
        return {
            "status": self.status(),
            "organization_id": "org-1",
            "counts": {
                "published": 1,
                "suggested": 2,
                "drafts": 3,
                "installed": 4,
                "notifications": 5,
            },
        }

    def search_skills(self, query=""):
        self.calls.append(("search_skills", query))
        return [
            {
                "id": "skill-1",
                "slug": "incident-handoff",
                "latest_version": 2,
                "author_description": "Summarize an incident",
            }
        ]

    def resolve_skill(self, reference, *, include_compatibility=True):
        self.calls.append(("resolve_skill", reference, include_compatibility))
        return {
            "skill": {
                "id": "skill-1",
                "slug": "incident-handoff",
            },
            "versions": [{"version": 2}, {"version": 1}],
            "local_compatibility": {"outcome": "compatible"},
            "latest_version_detail": {
                "version": {
                    "author_description": "Summarize an incident",
                    "scan": {"verdict": "pass"},
                    "verified_facts": {"scan_verdict": "pass"},
                    "system_spec": {
                        "hermes": {"minimum_version": "0.20.5"},
                        "platforms": ["macOS"],
                        "runtime": {"shell": True},
                    },
                }
            },
            "local_installation": {
                "version": 1,
                "update_mode": "MANUAL",
            },
        }

    def list_candidates(self, *, qualified_only=True, query=None):
        self.calls.append(("list_candidates", qualified_only, query))
        return [
            {
                "name": "incident-handoff",
                "editorial_name": "Incident Handoff",
                "editorial_description": "Transfer incident context between responders.",
                "local_skill_id": "local-1",
                "eligibility": "eligible",
                "qualification": "high_usage",
                "qualification_sequence": 1,
                "notice_variant": "first",
                "organization_name": "Nous Research",
            }
        ]

    def install_plan(self, reference, *, update_mode=None):
        self.calls.append(("install_plan", reference, update_mode))
        return {
            "receipt": "receipt-1",
            "skill_id": "skill-1",
            "slug": "incident-handoff",
            "version": 2,
            "compatibility": {"outcome": "compatible"},
            "allowed": True,
        }

    def install_apply(self, receipt, *, accept_partial=False):
        self.calls.append(("install_apply", receipt, accept_partial))
        return {"skill_id": "skill-1", "version": 2}

    def review(self, draft_id, *, acknowledge, portal=False):
        self.calls.append(("review", draft_id, acknowledge, portal))
        return {
            "draft": {
                "id": draft_id,
                "slug": "incident-handoff",
                "state": "ready",
                "authorDescription": "Summarize an incident",
                "scanVerdict": "pass",
                "scan": {"findings": []},
            },
            "hashes": {
                "content": "sha256:content",
                "author_description": "sha256:description",
                "package_manifest": "sha256:manifest",
            },
            "effective_policy": {"publication_mode": "moderated"},
        }

    def portal_review_url(self, draft_id):
        return f"https://portal.example/review/{draft_id}"

    def approve_owner_draft(self, draft_id):
        self.calls.append(("approve_owner_draft", draft_id))
        return {"publication_state": "pending_moderation"}

    def decline_owner_draft(self, draft_id):
        self.calls.append(("decline_owner_draft", draft_id))
        return {"state": "declined"}

    def list_installations(self):
        return [
            {
                "skill_id": "skill-1",
                "slug": "incident-handoff",
                "state": "active",
                "version": 1,
                "latest_version": 2,
                "update_mode": "MANUAL",
                "effective_update_mode": "AUTO_WITH_NOTICE",
                "skill_state": "active",
            }
        ]

    def update_all(self, *, apply=False):
        self.calls.append(("update_all", apply))
        return {
            "installations": [
                {
                    "skill_id": "skill-1",
                    "slug": "incident-handoff",
                    "state": "update_available",
                }
            ]
        }

    def update_plan(self, skill_id):
        self.calls.append(("update_plan", skill_id))
        return {
            "receipt": "update-receipt-1",
            "skill_id": skill_id,
            "slug": "incident-handoff",
            "version": 2,
            "compatibility": {"outcome": "compatible"},
            "allowed": True,
        }

    def update_apply(self, receipt):
        self.calls.append(("update_apply", receipt))
        return {"skill_id": "skill-1", "version": 2}

    def uninstall(self, skill_id):
        self.calls.append(("uninstall", skill_id))
        return {"skill_id": skill_id}

    def notifications(self, *, mark_seen=False):
        self.calls.append(("notifications", mark_seen))
        return {
            "events": []
            if mark_seen
            else [
                {
                    "message": "incident-handoff v2 is available",
                    "occurred_at": "2026-08-31T00:00:00Z",
                }
            ]
        }


@pytest.fixture(autouse=True)
def _fresh_callback_store(monkeypatch):
    monkeypatch.setattr(command_module, "CALLBACK_TOKENS", _CallbackTokens())


def test_parse_supports_quoted_search_and_cli_aliases():
    controller = WisdomCommandController()

    assert controller.parse('browse "incident handoff"') == (
        "browse",
        ["incident handoff"],
    )
    assert controller.parse("list deploy") == ("browse", ["deploy"])
    assert controller.parse("suggest local-skill") == (
        "submit",
        ["local-skill"],
    )


def test_candidates_uses_the_stable_first_notice_projection():
    service = _Service()

    view = WisdomCommandController().execute("candidates", service, _context())

    assert "Your organisation (Nous Research) has enabled Collective Wisdom" in (
        view.items[0].detail
    )
    assert "Congratulations! Hermes detected a skill" in view.items[0].detail
    assert "Why suggested: high usage" in view.items[0].detail
    assert view.items[0].title == "Incident Handoff"
    assert "Transfer incident context between responders." in view.items[0].detail


def test_skill_preview_keeps_install_as_the_trailing_action():
    view = WisdomCommandController().execute("show skill-1", _Service(), _context())

    assert [action.label for action in view.actions] == [
        "Versions",
        "View in Portal ↗",
        "Install",
    ]


def test_local_action_command_resumes_bound_controller_action():
    service = _Service()
    context = _context(chat_id="local:session-1")
    view = WisdomCommandController().execute("install skill-1", service, context)
    rendered = render_local_view(view, context)

    action_line = next(
        line for line in rendered.splitlines() if line.startswith("Manual:")
    )
    token = action_line.rsplit(" ", 1)[-1]
    resumed = WisdomCommandController().execute(f"action {token}", service, context)

    assert resumed.title == "Confirm install"
    assert ("install_plan", "skill-1", "MANUAL") in service.calls


def test_local_action_command_rejects_another_session():
    context = _context(chat_id="local:session-1")
    view = WisdomCommandController().execute("install skill-1", _Service(), context)
    render_local_view(view, context)
    token = view.actions[0].callback_data.removeprefix("wi:cmd:")

    with pytest.raises(PermissionError, match="another session"):
        WisdomCommandController().execute(
            f"action {token}",
            _Service(),
            _context(chat_id="local:session-2"),
        )


def test_unknown_keyword_returns_focused_help():
    view = WisdomCommandController().execute("wat", _Service(), _context())

    assert view.title == "Collective Wisdom commands"
    assert view.notice == "Unknown /wisdom keyword: wat"


def test_help_explains_every_workflow_and_includes_examples():
    view = WisdomCommandController().execute("help", _Service(), _context())
    rendered = view.to_text()

    assert [item.title for item in view.items] == [
        "Discover",
        "Contribute",
        "Manage installed skills",
        "Account and activity",
        "Examples",
    ]
    assert "/wisdom show <skill> — View its description" in rendered
    assert "/wisdom installed — List and manage skills installed" in rendered
    assert "/wisdom submit my-local-skill" in rendered


def test_group_home_does_not_read_private_counts():
    service = _Service()
    view = WisdomCommandController().execute(
        "", service, _context(is_group=True, chat_id="group-1")
    )

    assert service.calls == []
    assert [action.label for action in view.actions] == [
        "Browse",
        "Continue in DM",
        "Help",
    ]


def test_connected_home_includes_help_action():
    view = WisdomCommandController().execute("", _Service(), _context())

    help_action = next(action for action in view.actions if action.label == "Help")
    assert help_action.local_command == "/wisdom help"


def test_setup_disclosure_precedes_profile_mutation():
    service = _Service()
    service.command_home = lambda: {"status": {"configured": False}}
    controller = WisdomCommandController()
    context = _context()

    home = controller.execute("", service, context)
    bind_view_callbacks(home, context)
    setup_token = home.actions[0].callback_data.removeprefix("wi:cmd:")
    disclosure = controller.execute_token(setup_token, service, context)

    assert service.calls == []
    assert "Candidate qualification stays on this profile" in disclosure.summary
    bind_view_callbacks(disclosure, context)
    confirm_token = disclosure.actions[0].callback_data.removeprefix("wi:cmd:")
    controller.execute_token(confirm_token, service, context)
    assert service.calls == [("setup", True)]


def test_home_turns_authentication_failure_into_sign_in_card():
    service = _Service()
    status = service.status()
    status.update({"gateway_available": False, "error_kind": "authentication"})
    service.command_home = lambda: {"status": status}

    view = WisdomCommandController().execute("", service, _context())

    assert view.title == "Sign in to use Collective Wisdom"
    assert any(action.url for action in view.actions)


def test_status_reports_local_and_degraded_state_without_raw_error():
    service = _Service()
    status = service.status()
    status.update({
        "gateway_available": False,
        "error_kind": "unavailable",
        "error": "Authorization: Bearer secret-token",
        "pending_operations": [{"id": "one"}],
    })
    service.status = lambda: status

    view = WisdomCommandController().execute("status", service, _context())

    assert "Local store: ready · 1 pending operation(s)" in view.summary
    assert "State: Gateway unavailable" in view.summary
    assert "secret-token" not in view.summary


def test_group_show_removes_install_and_keeps_portal_link():
    service = _Service()
    view = WisdomCommandController().execute(
        "show skill-1", service, _context(is_group=True, chat_id="group-1")
    )

    assert "Install" not in [action.label for action in view.actions]
    assert "Continue in DM" in [action.label for action in view.actions]
    assert any(action.url for action in view.actions)
    assert ("resolve_skill", "skill-1", False) in service.calls


def test_group_versions_pagination_never_reintroduces_install_controls():
    service = _Service()
    original = service.resolve_skill

    def six_versions(reference, *, include_compatibility=True):
        value = original(reference, include_compatibility=include_compatibility)
        value["versions"] = [{"version": version} for version in range(6, 0, -1)]
        return value

    service.resolve_skill = six_versions
    controller = WisdomCommandController()
    context = _context(is_group=True, chat_id="group-1")
    first = controller.execute("versions skill-1", service, context)
    bind_view_callbacks(first, context)
    next_action = next(action for action in first.actions if action.label == "Next")

    second = controller.execute_token(
        next_action.callback_data.removeprefix("wi:cmd:"), service, context
    )

    assert all(
        action.label != "Install" for item in second.items for action in item.actions
    )
    assert any(action.label == "Continue in DM" for action in second.actions)


def test_private_show_includes_requirements_and_installation_state():
    view = WisdomCommandController().execute("show skill-1", _Service(), _context())

    assert "Summarize an incident" in view.summary
    assert "Latest: v2 · scan: pass" in view.summary
    assert "Requirements: Hermes 0.20.5+; OS: macOS; runtime: shell" in view.summary
    assert "Installed: v1 · MANUAL" in view.summary


def test_browse_navigation_can_step_back_through_show_and_versions():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()

    browse = controller.execute("browse", service, context)
    assert browse.navigation_actions == []
    bind_view_callbacks(browse, context)
    show_action = browse.items[0].actions[0]
    show = controller.execute_token(
        show_action.callback_data.removeprefix("wi:cmd:"), service, context
    )

    assert [action.label for action in show.navigation_actions] == ["← Back"]
    bind_view_callbacks(show, context)
    versions_action = next(
        action for action in show.actions if action.operation == "versions"
    )
    versions = controller.execute_token(
        versions_action.callback_data.removeprefix("wi:cmd:"), service, context
    )

    assert versions.title == "incident-handoff versions"
    bind_view_callbacks(versions, context)
    back_to_show = controller.execute_token(
        versions.navigation_actions[0].callback_data.removeprefix("wi:cmd:"),
        service,
        context,
    )

    assert back_to_show.title == "incident-handoff"
    bind_view_callbacks(back_to_show, context)
    back_to_browse = controller.execute_token(
        back_to_show.navigation_actions[0].callback_data.removeprefix("wi:cmd:"),
        service,
        context,
    )

    assert back_to_browse.title == "Shared skills"
    assert back_to_browse.navigation_actions == []


def test_review_shows_exact_hashes_scan_and_effective_policy():
    service = _Service()
    view = WisdomCommandController().execute("review draft-1", service, _context())

    assert "Content: sha256:content" in view.summary
    assert "Description: sha256:description" in view.summary
    assert "Manifest: sha256:manifest" in view.summary
    assert "Server scan: pass" in view.summary
    assert "Publication policy: moderated" in view.summary


def test_publish_callback_uses_race_safe_service_operation():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    review = controller.execute("review draft-1", service, context)
    bind_view_callbacks(review, context)
    publish = next(action for action in review.actions if action.operation == "publish")

    result = controller.execute_token(
        publish.callback_data.removeprefix("wi:cmd:"), service, context
    )

    assert ("approve_owner_draft", "draft-1") in service.calls
    assert result.summary == "Sent to your collective administrator for approval."


def test_installed_view_shows_installed_latest_and_effective_mode():
    view = WisdomCommandController().execute("installed", _Service(), _context())

    assert view.items[0].detail == ("installed v1 · latest v2 · AUTO_WITH_NOTICE")


def test_install_requires_mode_plan_and_second_confirmation():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    modes = controller.execute("install skill-1", service, context)
    bind_view_callbacks(modes, context)

    assert not service.calls
    plan_token = modes.actions[0].callback_data.removeprefix("wi:cmd:")
    plan = controller.execute_token(plan_token, service, context)
    bind_view_callbacks(plan, context)
    assert service.calls == [("install_plan", "skill-1", None)]
    assert not any(call[0] == "install_apply" for call in service.calls)

    apply_token = plan.actions[0].callback_data.removeprefix("wi:cmd:")
    controller.execute_token(apply_token, service, context)
    assert service.calls[-1] == ("install_apply", "receipt-1", False)


def test_install_preview_can_step_back_to_modes_and_skill():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    show = controller.execute("show skill-1", service, context)
    bind_view_callbacks(show, context)

    install = next(
        action for action in show.actions if action.operation == "install_modes"
    )
    modes = controller.execute_token(
        install.callback_data.removeprefix("wi:cmd:"), service, context
    )
    bind_view_callbacks(modes, context)
    manual = next(
        action
        for action in modes.actions
        if action.arguments.get("update_mode") == "MANUAL"
    )
    plan = controller.execute_token(
        manual.callback_data.removeprefix("wi:cmd:"), service, context
    )
    bind_view_callbacks(plan, context)

    back_to_modes = controller.execute_token(
        plan.navigation_actions[0].callback_data.removeprefix("wi:cmd:"),
        service,
        context,
    )
    assert back_to_modes.title == "Choose how future updates are handled"
    bind_view_callbacks(back_to_modes, context)

    back_to_skill = controller.execute_token(
        back_to_modes.navigation_actions[0].callback_data.removeprefix("wi:cmd:"),
        service,
        context,
    )
    assert back_to_skill.title == "incident-handoff"


def test_blocked_install_plan_never_offers_quick_application():
    service = _Service()
    service.install_plan = lambda *_args, **_kwargs: {
        "receipt": "blocked-receipt",
        "skill_id": "skill-1",
        "slug": "incident-handoff",
        "version": 2,
        "compatibility": {"outcome": "blocked_pending_action"},
        "allowed": False,
    }
    controller = WisdomCommandController()
    context = _context()
    modes = controller.execute("install skill-1", service, context)
    bind_view_callbacks(modes, context)

    plan = controller.execute_token(
        modes.actions[0].callback_data.removeprefix("wi:cmd:"), service, context
    )

    assert plan.actions == []
    assert "full compatibility review" in str(plan.notice)


def test_update_all_only_presents_eligible_updates():
    service = _Service()

    view = WisdomCommandController().execute("update all", service, _context())

    assert ("update_all", False) in service.calls
    assert not any(call[0] == "update_apply" for call in service.calls)
    assert view.items[0].actions[0].label == "Review update"


def test_uninstall_requires_callback_confirmation_before_mutation():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()

    confirmation = controller.execute("uninstall incident-handoff", service, context)

    assert not any(call[0] == "uninstall" for call in service.calls)
    bind_view_callbacks(confirmation, context)
    token = confirmation.actions[0].callback_data.removeprefix("wi:cmd:")
    result = controller.execute_token(token, service, context)
    assert ("uninstall", "skill-1") in service.calls
    assert result.title == "Skill uninstalled"


def test_notifications_are_not_marked_read_until_confirmation():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()

    view = controller.execute("notifications", service, context)

    assert service.calls == [("notifications", False)]
    bind_view_callbacks(view, context)
    token = view.actions[0].callback_data.removeprefix("wi:cmd:")
    controller.execute_token(token, service, context)
    assert service.calls[-1] == ("notifications", True)


def test_callback_is_short_scoped_and_single_use_after_success():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    view = controller.execute("install skill-1", service, context)
    bind_view_callbacks(view, context)
    callback_data = view.actions[0].callback_data
    token = callback_data.removeprefix("wi:cmd:")

    assert len(callback_data.encode("utf-8")) <= 64
    with pytest.raises(PermissionError):
        controller.execute_token(token, service, _context(user_id="user-2"))
    with pytest.raises(PermissionError):
        controller.execute_token(token, service, _context(chat_id="chat-2"))
    with pytest.raises(PermissionError):
        controller.execute_token(token, service, _context(profile="other"))
    with pytest.raises(PermissionError):
        controller.execute_token(token, service, _context(organization_id="org-2"))

    plan = controller.execute_token(token, service, context)
    bind_view_callbacks(plan, context)
    mutation_token = plan.actions[0].callback_data.removeprefix("wi:cmd:")
    controller.execute_token(mutation_token, service, context)
    with pytest.raises(ValueError, match="expired"):
        controller.execute_token(mutation_token, service, context)


def test_transient_mutation_failure_keeps_callback_retryable():
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    view = controller.execute("install skill-1", service, context)
    bind_view_callbacks(view, context)
    token = view.actions[0].callback_data.removeprefix("wi:cmd:")
    original = service.install_plan
    service.install_plan = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("temporary")
    )

    with pytest.raises(RuntimeError, match="temporary"):
        controller.execute_token(token, service, context)

    service.install_plan = original
    assert controller.execute_token(token, service, context).title == "Confirm install"


def test_expired_callback_is_rejected(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(command_module.time, "monotonic", lambda: now[0])
    service = _Service()
    controller = WisdomCommandController()
    context = _context()
    view = controller.execute("install skill-1", service, context)
    bind_view_callbacks(view, context)
    token = view.actions[0].callback_data.removeprefix("wi:cmd:")
    now[0] = 800.0

    with pytest.raises(ValueError, match="expired"):
        controller.execute_token(token, service, context)


def test_group_continuation_is_user_profile_and_org_bound():
    group = _context(chat_id="group-1", is_group=True)
    token = issue_continuation("drafts", group)

    with pytest.raises(PermissionError):
        resolve_continuation(token, _context(user_id="user-2", chat_id="dm-2"))
    with pytest.raises(PermissionError):
        resolve_continuation(token, _context(profile="other", chat_id="dm-1"))
    with pytest.raises(PermissionError):
        resolve_continuation(
            token,
            _context(organization_id="org-2", chat_id="dm-1"),
        )

    assert resolve_continuation(token, _context(chat_id="dm-1")) == "drafts"
    with pytest.raises(ValueError, match="expired"):
        resolve_continuation(token, _context(chat_id="dm-1"))


def test_group_continuation_cannot_be_redeemed_in_another_group():
    token = issue_continuation(
        "installed",
        _context(chat_id="group-1", is_group=True),
    )

    with pytest.raises(PermissionError):
        resolve_continuation(
            token,
            _context(chat_id="group-2", is_group=True),
        )


def test_unexpected_errors_are_not_echoed_to_text_fallbacks():
    assert "secret-token" not in command_error_text(
        RuntimeError("Authorization: Bearer secret-token")
    )


def test_text_fallback_keeps_portal_links_but_not_callback_payloads():
    view = command_module.WisdomView(
        "Collective Wisdom",
        items=[
            command_module.WisdomItem(
                "incident-handoff",
                actions=[
                    command_module.WisdomAction(
                        "View in Portal", url="https://portal.example/skill-1"
                    ),
                    command_module.WisdomAction(
                        "Install", callback_data="wi:cmd:private-token"
                    ),
                ],
            )
        ],
    )

    rendered = view.to_text()

    assert "View in Portal: https://portal.example/skill-1" in rendered
    assert "private-token" not in rendered


def test_local_text_prefers_readable_commands_without_changing_shared_fallback():
    context = _context(chat_id="local:session-1")
    view = command_module.WisdomView(
        "Collective Wisdom",
        actions=[
            command_module.WisdomAction(
                "Browse", "browse", local_command="/wisdom browse"
            )
        ],
    )

    rendered = render_local_view(view, context)

    assert "Browse: /wisdom browse" in rendered
    assert "/wisdom action" not in rendered
    assert "wi:cmd:" not in rendered
    assert view.to_text() == "Collective Wisdom"


def test_local_browse_renders_show_by_skill_name_instead_of_action_token():
    view = WisdomCommandController().execute("browse", _Service(), _context())

    rendered = render_local_view(view, _context())

    assert "View: /wisdom show incident-handoff" in rendered
    assert "/wisdom action" not in rendered
