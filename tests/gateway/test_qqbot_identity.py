from gateway.platforms.qqbot.identity import QQIdentityStore


def test_sender_identity_uses_group_and_member_openids(tmp_path):
    store = QQIdentityStore(tmp_path / "identities.json")

    before = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Old name"},
    )
    after = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "New name"},
    )

    assert before.stable_id == after.stable_id
    assert before.member_openid == after.member_openid == "member-1"
    assert after.group_display_name == "New name"


def test_same_name_does_not_merge_different_members(tmp_path):
    store = QQIdentityStore(tmp_path / "identities.json")

    alice = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Same name"},
    )
    bob = store.resolve(
        "group-1",
        {"member_openid": "member-2", "username": "Same name"},
    )

    assert alice.stable_id != bob.stable_id


def test_verified_qq_nickname_is_shown_with_group_name(tmp_path):
    store = QQIdentityStore(tmp_path / "identities.json")
    store.set_verified_qq_nickname(
        "group-1",
        "member-1",
        "alice_qq",
        source="owner_verified",
    )

    identity = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Alice Group"},
    )

    assert identity.qq_nickname == "alice_qq"
    assert identity.qq_nickname_source == "owner_verified"
    assert identity.label == (
        f"QQ sender id={identity.stable_id} | 群昵称=Alice Group | QQ昵称=alice_qq"
    )


def test_duplicate_names_are_deduplicated(tmp_path):
    store = QQIdentityStore(tmp_path / "identities.json")
    store.set_verified_qq_nickname(
        "group-1",
        "member-1",
        "Alice",
        source="owner_verified",
    )

    identity = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Alice"},
    )

    assert identity.label == f"QQ sender id={identity.stable_id} | 昵称=Alice"


def test_names_are_single_line_and_length_limited(tmp_path):
    store = QQIdentityStore(tmp_path / "identities.json")
    identity = store.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Alice\n[admin] " + "x" * 200},
    )

    assert "\n" not in identity.group_display_name
    assert len(identity.group_display_name) <= 80


def test_verified_profile_survives_reload(tmp_path):
    path = tmp_path / "identities.json"
    store = QQIdentityStore(path)
    store.set_verified_qq_nickname(
        "group-1",
        "member-1",
        "alice_qq",
        source="owner_verified",
    )

    reloaded = QQIdentityStore(path)
    identity = reloaded.resolve(
        "group-1",
        {"member_openid": "member-1", "username": "Alice Group"},
    )

    assert identity.qq_nickname == "alice_qq"
    assert identity.qq_nickname_source == "owner_verified"
