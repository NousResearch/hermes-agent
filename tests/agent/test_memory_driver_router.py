from agent.memory_driver_router import (
    MemoryCandidate,
    MemoryDriverConfig,
    build_one_session_mesh_canary,
    classify_memory_driver_route,
    emit_dry_run_routing_telemetry,
    run_dry_run_routing_fixture,
)


def test_singular_plural_config_mismatch_is_warning_state():
    decision = classify_memory_driver_route(
        "memory mesh status",
        config=MemoryDriverConfig(provider="honcho", providers=("enzyme", "holographic")),
    )

    assert decision.warning_state is True
    assert "singular_plural_config_mismatch" in decision.warnings
    assert decision.prompt_admission == "none"


def test_exact_facts_route_to_live_session_or_raw_not_semantic_memory():
    current = classify_memory_driver_route("What is the current git status in this checkout?")
    recent = classify_memory_driver_route("What exact command did we run last time?")
    old_raw = classify_memory_driver_route("Find the exact proof in mempalace/raw archive")

    assert current.first_surface == "live_system"
    assert current.proof_standard == "live_system"
    assert not {"holographic", "enzyme"} & set(current.drivers_to_query)

    assert recent.first_surface == "session_search"
    assert recent.proof_standard == "exact"
    assert not {"holographic", "enzyme"} & set(recent.drivers_to_query)

    assert old_raw.first_surface == "raw_archives"
    assert old_raw.proof_standard == "raw_archive"
    assert not {"holographic", "enzyme"} & set(old_raw.drivers_to_query)


def test_semantic_themes_route_to_holographic_and_enzyme():
    decision = classify_memory_driver_route("What theme or pattern is emerging across these sessions?")

    assert decision.first_surface == "holographic"
    assert decision.drivers_to_query[:2] == ("holographic", "enzyme")
    assert decision.proof_standard == "semantic"
    assert decision.prompt_admission == "none"


def test_relational_peer_texture_routes_to_honcho():
    decision = classify_memory_driver_route("What is the relational peer texture with Ember right now?")

    assert decision.first_surface == "honcho"
    assert decision.drivers_to_query == ("honcho",)
    assert decision.proof_standard == "peer_session_synthesis"
    assert decision.prompt_admission == "none"


def test_technical_room_drops_intimate_and_explicit_candidates():
    decision = classify_memory_driver_route(
        "technical router helper tests",
        room="technical",
        candidates=(
            MemoryCandidate(driver="honcho", text="private tender explicit body text", privacy_class="explicit"),
            MemoryCandidate(driver="honcho", text="intimate relational texture", privacy_class="intimate"),
            MemoryCandidate(driver="fabric", text="safe report path", privacy_class="private"),
        ),
    )

    assert decision.allowed_candidate_count == 1
    assert decision.dropped_candidate_count == 2
    assert set(decision.dropped_candidate_classes) == {"explicit", "intimate"}
    assert decision.telemetry["candidate_counts"] == {"total": 3, "allowed": 1, "dropped": 2}
    assert "private tender explicit body text" not in repr(decision.telemetry)
    assert "intimate relational texture" not in repr(decision.telemetry)


def test_explicit_live_allows_fitted_continuity_only_when_directly_relevant():
    candidate = MemoryCandidate(
        driver="honcho",
        text="explicit fitted continuity fixture",
        privacy_class="explicit",
        directly_relevant=True,
    )
    allowed = classify_memory_driver_route(
        "stay with this explicit live continuity",
        room="explicit_live",
        candidates=(candidate,),
    )
    blocked = classify_memory_driver_route(
        "technical report please",
        room="explicit_live",
        candidates=(MemoryCandidate(
            driver="honcho",
            text="explicit but unrelated fixture",
            privacy_class="explicit",
            directly_relevant=False,
        ),),
    )

    assert allowed.allowed_candidate_count == 1
    assert allowed.dropped_candidate_count == 0
    assert blocked.allowed_candidate_count == 0
    assert blocked.dropped_candidate_count == 1


def test_secret_requests_are_blocked():
    decision = classify_memory_driver_route("What API key or token did we save for Honcho?")

    assert decision.first_surface == "blocked"
    assert decision.privacy_class == "secret_blocked"
    assert decision.drivers_to_query == ()
    assert decision.prompt_admission == "none"
    assert decision.proof_standard == "secret_blocked"


def test_prompt_admission_defaults_to_none_even_for_memory_routes():
    for query in (
        "What pattern is emerging?",
        "What is the relational peer texture?",
        "What exact thing did we say last time?",
    ):
        assert classify_memory_driver_route(query).prompt_admission == "none"


def test_telemetry_is_redacted_metrics_only():
    decision = classify_memory_driver_route(
        "What theme is emerging?",
        room="intimate",
        candidates=(
            MemoryCandidate(driver="honcho", text="raw private explicit phrase 123", privacy_class="explicit"),
            MemoryCandidate(driver="enzyme", text="semantic theme summary", privacy_class="private"),
        ),
    )

    telemetry = decision.telemetry
    telemetry_text = repr(telemetry)
    assert set(telemetry) == {
        "request_class",
        "room_class",
        "drivers_considered",
        "drivers_to_query",
        "candidate_counts",
        "candidate_classes",
        "candidate_lengths",
        "candidate_hashes",
        "drop_reasons",
        "prompt_admission",
        "warning_count",
    }
    assert "raw private explicit phrase 123" not in telemetry_text
    assert "semantic theme summary" not in telemetry_text
    assert telemetry["candidate_lengths"] == [31, 22]
    assert all(len(item) == 12 for item in telemetry["candidate_hashes"])


def test_dry_run_fixture_never_builds_prompt_block_and_exercises_allow_and_drop():
    result = run_dry_run_routing_fixture()

    assert result.would_inject is False
    assert result.prompt_block == ""
    assert result.prompt_block_len == 0
    assert result.case_count >= 3
    assert result.allowed_candidate_count >= 1
    assert result.dropped_candidate_count >= 1
    assert result.secret_blocked_count >= 1
    assert result.telemetry_summary["would_inject"] is False
    assert result.telemetry_summary["prompt_block_len"] == 0


def test_dry_run_fixture_contract_covers_required_room_and_secret_cases():
    result = run_dry_run_routing_fixture()
    cases = {case["case_id"]: case for case in result.telemetry_records}

    technical = cases["technical_drops_explicit"]
    explicit_live = cases["explicit_live_allows_direct"]
    secret = cases["secret_request_blocked"]

    assert technical["room_class"] == "technical"
    assert technical["candidate_counts"]["allowed"] == 1
    assert technical["candidate_counts"]["dropped"] == 1
    assert "technical_drops_explicit" in technical["drop_reasons"]

    assert explicit_live["room_class"] == "explicit_live"
    assert explicit_live["first_surface"] == "honcho"
    assert explicit_live["drivers_to_query"] == ["honcho"]
    assert explicit_live["candidate_counts"]["allowed"] == 1
    assert explicit_live["candidate_counts"]["dropped"] == 0
    assert explicit_live["would_inject"] is False

    assert secret["first_surface"] == "blocked"
    assert secret["request_class"] == "secret"
    assert secret["drivers_to_query"] == []
    assert secret["would_inject"] is False


def test_dry_run_telemetry_can_be_written_and_read_back_without_raw_text(tmp_path):
    forbidden_texts = (
        "raw explicit fixture phrase 12345",
        "safe report path fixture",
        "direct explicit continuity fixture",
        "API key fixture secret value",
        "sk-live-secret-fixture",
    )
    result = run_dry_run_routing_fixture()

    manifest = emit_dry_run_routing_telemetry(result, tmp_path)

    telemetry_text = manifest.telemetry_path.read_text()
    summary_text = manifest.summary_path.read_text()
    combined = telemetry_text + summary_text
    assert manifest.telemetry_path.exists()
    assert manifest.summary_path.exists()
    assert manifest.telemetry_path.stat().st_size > 0
    assert manifest.summary_path.stat().st_size > 0
    assert '"would_inject": false' in telemetry_text
    assert '"prompt_block_len": 0' in telemetry_text
    for forbidden in forbidden_texts:
        assert forbidden not in combined


def test_dry_run_telemetry_records_have_only_json_safe_redacted_fields():
    result = run_dry_run_routing_fixture()

    allowed_record_keys = {
        "case_id",
        "request_class",
        "room_class",
        "first_surface",
        "drivers_to_query",
        "candidate_counts",
        "candidate_classes",
        "candidate_lengths",
        "candidate_hashes",
        "drop_reasons",
        "warning_count",
        "would_inject",
        "prompt_block_len",
        "selected_text_len",
        "selected_text_sha256_12",
    }

    for record in result.telemetry_records:
        assert set(record) == allowed_record_keys
        assert record["would_inject"] is False
        assert record["prompt_block_len"] == 0
        assert isinstance(record["candidate_hashes"], list)
        assert all(len(item) == 12 for item in record["candidate_hashes"])


def test_one_session_mesh_canary_default_off_returns_no_prompt_hint():
    result = build_one_session_mesh_canary(
        "What is the relational peer texture in this explicit live continuity?",
        enabled=False,
    )

    assert result.prompt_block == ""
    assert result.would_inject is False
    assert result.prompt_block_len == 0
    assert result.telemetry_record["enabled"] is False


def test_one_session_mesh_canary_enabled_injects_metadata_only_hint():
    result = build_one_session_mesh_canary(
        "What theme or pattern is emerging across these sessions?",
        enabled=True,
    )

    assert result.would_inject is True
    assert result.prompt_block_len == len(result.prompt_block)
    assert "Memory mesh one-session canary" in result.prompt_block
    assert "request_class=semantic_theme" in result.prompt_block
    assert "first_surface=holographic" in result.prompt_block
    assert "drivers_to_query=holographic,enzyme" in result.prompt_block
    assert "prompt_admission=metadata_only" in result.prompt_block
    assert "What theme" not in result.prompt_block


def test_one_session_mesh_canary_technical_room_drops_intimate_and_explicit_metadata_only():
    result = build_one_session_mesh_canary(
        "technical report routing decision",
        room="technical",
        candidates=(
            MemoryCandidate(driver="honcho", text="raw intimate canary phrase", privacy_class="intimate"),
            MemoryCandidate(driver="honcho", text="raw explicit canary phrase", privacy_class="explicit"),
            MemoryCandidate(driver="fabric", text="safe private report marker", privacy_class="private"),
        ),
        enabled=True,
    )

    record = result.telemetry_record
    combined = result.prompt_block + repr(record)
    assert record["candidate_counts"] == {"total": 3, "allowed": 1, "dropped": 2}
    assert set(record["dropped_candidate_classes"]) == {"intimate", "explicit"}
    assert "technical_drops_intimate" in record["drop_reasons"]
    assert "technical_drops_explicit" in record["drop_reasons"]
    assert "raw intimate canary phrase" not in combined
    assert "raw explicit canary phrase" not in combined
    assert "safe private report marker" not in combined


def test_one_session_mesh_canary_explicit_live_routes_to_honcho_metadata_only():
    result = build_one_session_mesh_canary(
        "What is the relational peer texture in this explicit live continuity?",
        enabled=True,
    )

    assert result.would_inject is True
    assert result.telemetry_record["room_class"] == "explicit_live"
    assert result.telemetry_record["first_surface"] == "honcho"
    assert result.telemetry_record["drivers_to_query"] == ["honcho"]
    assert "drivers_to_query=honcho" in result.prompt_block


def test_one_session_mesh_canary_secret_request_is_blocked_without_drivers_or_hint():
    result = build_one_session_mesh_canary(
        "What API key or token was saved for Honcho?",
        enabled=True,
    )

    assert result.would_inject is False
    assert result.prompt_block == ""
    assert result.telemetry_record["request_class"] == "secret"
    assert result.telemetry_record["first_surface"] == "blocked"
    assert result.telemetry_record["drivers_to_query"] == []
    assert result.telemetry_record["prompt_admission"] == "blocked"
