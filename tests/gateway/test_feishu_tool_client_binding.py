"""Tests for Feishu adapter client binding into Feishu document tools."""

from unittest.mock import patch


def test_set_tool_clients_updates_shared_feishu_tool_clients():
    from gateway.platforms.feishu import FeishuAdapter

    client = object()
    with patch("tools.feishu_doc_tool.set_shared_client") as set_doc_client, patch(
        "tools.feishu_drive_tool.set_shared_client"
    ) as set_drive_client:
        FeishuAdapter._set_tool_clients(client)

    set_doc_client.assert_called_once_with(client)
    set_drive_client.assert_called_once_with(client)
