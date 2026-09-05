"""Files copy uses real catalogs; visible versions keep their exact actions."""

from datetime import datetime, timezone
from string import Formatter

import pytest

from agent import i18n
from gateway import hosted_room_messaging_files as files
from tests.gateway.test_hosted_room_messaging_files import consumer, event
from tests.gateway.test_hosted_room_file_access import file_state, publish


@pytest.fixture(autouse=True)
def language(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


async def menu_with_versions(consumer):
    state, runner, _ = consumer
    publish(state, "same.md", b"first")
    publish(state, "same.md", b"later")
    menu = files.FilesMenu(runner, event("/group 1 files"), state.backend, "/group")
    await menu.bind("1")
    await menu.files_page()
    return menu


@pytest.mark.asyncio
async def test_french_native_and_plain_copy_keep_command_values(consumer, monkeypatch):
    menu = await menu_with_versions(consumer)
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    page = menu.render_files()
    assert page.title.startswith("Fichiers")
    assert "Rechercher" in [choice["label"] for choice in page.choices]
    plain = menu.plain_files()
    assert "Obtenir la réponse complète:" in plain
    assert "`/group 1 files <text>`" in plain
    assert "`/group 1 reply`" in plain


def test_error_copy_uses_real_french_catalog(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    assert "vérifié" in files._error(ValueError("file_integrity_failed"))
    assert "Desktop" in files._error(ValueError("classic_files_on_desktop"))


@pytest.mark.asyncio
async def test_same_minute_versions_have_distinct_native_and_plain_labels(consumer):
    menu = await menu_with_versions(consumer)
    items = menu.pages[0]["items"]
    instant = datetime(2026, 9, 4, 8, 12, tzinfo=timezone.utc).timestamp()
    for item, seconds in zip(items, [7, 42]):
        item["shared_at"] = instant + seconds
    page = menu.render_files()
    captions = [choice["label"] for choice in page.choices[:2]]
    assert len(set(captions)) == 2
    assert "08:12:07" in captions[0] and "08:12:42" in captions[1]
    assert all(caption in menu.plain_files() for caption in captions)
    for choice, item in zip(page.choices, items):
        assert menu.actions[choice["value"]][1][0]["attachment_id"] == item["attachment_id"]


@pytest.mark.parametrize("lang", i18n.SUPPORTED_LANGUAGES)
def test_all_files_catalogs_render_values_without_fallback_or_missing_fields(monkeypatch, lang):
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    english = i18n._load_catalog("en")
    catalog = i18n._load_catalog(lang)
    keys = {key for key in english if key.startswith("gateway.group_files.")}
    assert keys
    values = {
        "name": files.label("/unsafe\u202e @all **name**"),
        "producer": files.label("[author](url)"), "date": "2026-09-04 08:12",
        "size": "11.0 MB", "caption": "caption", "command": "`/group 1 file abcdef12 confirm`",
        "files": "`/group 1 files`", "file": "`file <file-id>`", "reply": "`reply`",
    }
    for key in keys:
        assert key in catalog
        fields = {field for _, field, _, _ in Formatter().parse(catalog[key]) if field}
        assert fields == {field for _, field, _, _ in Formatter().parse(english[key]) if field}
        rendered = i18n.t(key, **{field: values[field] for field in fields})
        assert not rendered.startswith("gateway.group_files.")
        for field in fields:
            assert values[field] in rendered
    assert files._error(ValueError("file_access_denied")) == i18n.t("gateway.group_files.denied")
    assert files._status_text("delivered", "failed") == i18n.t("gateway.group_files.delivered")
    assert files._status_text("not-a-code", "failed") == i18n.t("gateway.group_files.failed")


@pytest.mark.asyncio
async def test_localized_confirmation_and_help_keep_unsafe_names_and_commands_separate(consumer, monkeypatch):
    menu = await menu_with_versions(consumer)
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    item = {**menu.pages[0]["items"][0], "name": "/group 7 stop\u202e @all **{name}**", "size": 11_000_000}
    page = await menu.prepare_file(item)
    assert page.title.startswith("Envoyer ")
    assert files.label(item["name"]) in page.title
    assert "\u202e" not in page.title and "@" not in page.title and "{name}" not in page.title
    assert [choice["label"] for choice in page.choices] == ["Envoyer", "Annuler"]
    selected = menu.actions[page.choices[0]["value"]]
    assert selected == ("file", (item, True, ""))
    help_text = menu.runner._group_chat_help("!group")
    assert "`!group 7 files [query]` - Rechercher" in help_text
    assert "`!group 7 file <file-id>` - Récupérer" in help_text
    assert "`!group 7 reply` - Récupérer" in help_text
    assert "`!group 7 stop` - Stop the current work." in help_text


@pytest.mark.asyncio
@pytest.mark.parametrize("spacing", [0.125, 0])
async def test_same_second_and_batch_timestamp_labels_survive_native_limits(consumer, spacing):
    from gateway.choice_picker import choice_label

    menu = await menu_with_versions(consumer)
    items = menu.pages[0]["items"]
    for index, item in enumerate(items):
        item["name"] = "long-name-" * 20
        item["producer"]["label"] = "long-producer-" * 10
        item["shared_at"] = 1_788_509_527 + index * spacing
    page = menu.render_files()
    captions = [choice_label(choice) for choice in page.choices[:2]]
    assert len(set(captions)) == 2
    assert all(len(caption) <= 100 for caption in captions)
    assert all(caption in menu.plain_files() for caption in captions)


@pytest.mark.asyncio
async def test_existing_timezone_resolution_and_dst_fold_are_respected(consumer, monkeypatch):
    menu = await menu_with_versions(consumer)
    monkeypatch.setenv("HERMES_TIMEZONE", "Europe/Zurich")
    items = menu.pages[0]["items"]
    for item, hour in zip(items, [0, 1]):
        item["shared_at"] = datetime(2026, 10, 25, hour, 30, tzinfo=timezone.utc).timestamp()
    captions = [menu.file_label(item) for item in items]
    assert all("02:30" in caption for caption in captions)
    assert "+0200" in captions[0] and "+0100" in captions[1]
    assert captions[0] != captions[1]


@pytest.mark.asyncio
async def test_unambiguous_labels_keep_minute_precision_and_actions_do_not_translate(consumer, monkeypatch):
    menu = await menu_with_versions(consumer)
    for index, item in enumerate(menu.pages[0]["items"]):
        item["shared_at"] = datetime(2026, 9, 4, 8, index, 12, tzinfo=timezone.utc).timestamp()
    english = menu.render_files()
    actions = list(menu.actions.values())
    assert "08:00:12" not in english.choices[0]["label"]
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    german = menu.render_files()
    assert list(menu.actions.values()) == actions
    assert german.title.startswith("Dateien")
    assert "04.09.2026 08:00" in german.choices[0]["label"]


@pytest.mark.asyncio
async def test_localized_native_delivery_keeps_original_bytes_and_filename(consumer, monkeypatch):
    from gateway.choice_picker import ChoiceProgress
    from tests.gateway.test_hosted_room_messaging_files import open_files

    state, runner, adapter = consumer
    publish(state, "document.txt", b"original bytes {not a translation}")
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    callback, page = await open_files(runner, adapter)
    progress = await callback("chat", page.choices[0]["value"])
    assert isinstance(progress, ChoiceProgress)
    sent = await progress.complete()
    assert sent.title == "Envoyé."
    assert adapter.documents[0][1:3] == (b"original bytes {not a translation}", "document.txt")


def test_files_copy_resolves_temporary_config_language(monkeypatch):
    from hermes_cli.config import save_config

    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    save_config({"display": {"language": "de"}})
    i18n.reset_language_cache()
    assert files.text("files") == "Dateien ansehen"
    assert files.text("full_reply") == "Vollständige Antwort abrufen"


@pytest.mark.asyncio
async def test_exact_timestamp_disambiguation_extends_existing_colliding_codes(consumer, monkeypatch):
    from gateway import hosted_room_file_lookup as lookup

    menu = await menu_with_versions(consumer)
    items = menu.pages[0]["items"]
    for item in items:
        item["shared_at"] = 1_788_509_527
    codes = {item["attachment_id"]: "deadbeef" + str(index) * 56 for index, item in enumerate(items)}
    monkeypatch.setattr(lookup, "selection_digest", lambda room, item: codes[item["attachment_id"]])
    captions = [menu.file_label(item) for item in items]
    assert "deadbeef0000" in captions[0] and "deadbeef1111" in captions[1]
    assert all(len(caption) <= 100 for caption in captions)
