"""Compound extensions (``.blade.php``) route to their own server; the last segment alone keeps its server."""
from agent.lsp.servers import ServerContext, find_server_for_file, language_id_for


def test_blade_templates_route_to_laravel_lsp_and_plain_php_stays_on_intelephense(monkeypatch, tmp_path):
    blade = "/app/resources/views/Home.BLADE.php"
    assert find_server_for_file(blade).server_id == "laravel-lsp"
    assert language_id_for(blade) == "blade"
    assert find_server_for_file("/app/app/Models/User.php").server_id == "intelephense"
    assert language_id_for("/app/app/Models/User.php") == "php"
    # Manual-install server: spawns `laravel-lsp lsp` only when the binary resolves; None otherwise.
    srv = find_server_for_file(blade)
    monkeypatch.setattr("agent.lsp.servers._which", lambda *names: None)
    assert srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path), install_strategy="manual")) is None
    monkeypatch.setattr("agent.lsp.servers._which", lambda *names: "/usr/local/bin/laravel-lsp")
    spec = srv.build_spawn(str(tmp_path), ServerContext(str(tmp_path), install_strategy="manual"))
    assert spec.command == ["/usr/local/bin/laravel-lsp", "lsp"]
