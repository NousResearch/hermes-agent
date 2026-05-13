import json

import yaml

from hermes_cli.paperclip_runtime import paperclip_runtime_diagnostics


def test_paperclip_runtime_diagnostics_reports_context_contract_hash_and_unmanaged_cron(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / ".hermes"
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir(parents=True)
    contracts = hermes_home / "routine-contracts.yaml"
    contracts.write_text(
        yaml.safe_dump(
            {
                "routines": [
                    {
                        "routineKey": "ijt-capital.coo.daily-operating-brief",
                        "title": "Daily Operating Brief",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (cron_dir / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "backed",
                    "name": "Backed Paperclip routine",
                    "metadata": {
                        "paperclip_routine_key": "ijt-capital.coo.daily-operating-brief",
                    },
                },
                {
                    "id": "local-only",
                    "name": "Legacy local report",
                    "enabled": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPERCLIP_MANAGED_ROUTINES", "true")
    monkeypatch.setenv("FLEET_RUNTIME_ID", "raava-ijt-capital-aurum-coo")
    monkeypatch.setenv("PAPERCLIP_COMPANY_ID", "pc-co-ijt")
    monkeypatch.setenv("PAPERCLIP_NODE_ID", "pc-agent-coo")
    monkeypatch.setenv("PAPERCLIP_TITLE", "COO")

    result = paperclip_runtime_diagnostics(hermes_home=hermes_home)

    assert result["paperclip_managed_routines"] is True
    assert result["context"]["fleet_runtime_id"] == "raava-ijt-capital-aurum-coo"
    assert result["context"]["paperclip_node_id"] == "pc-agent-coo"
    assert result["routine_contract"]["exists"] is True
    assert result["routine_contract"]["routine_keys"] == [
        "ijt-capital.coo.daily-operating-brief"
    ]
    assert result["local_cron"]["not_backed_by_paperclip_routines"] == [
        {
            "id": "local-only",
            "name": "Legacy local report",
            "enabled": True,
            "paperclip_routine_key": None,
        }
    ]
