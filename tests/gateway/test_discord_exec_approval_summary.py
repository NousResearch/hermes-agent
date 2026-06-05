from plugins.platforms.discord.adapter import _summarize_exec_approval


def test_summarize_exec_approval_explains_delete_in_japanese():
    text = _summarize_exec_approval("rm -rf /tmp/demo", "recursive delete")

    assert "何をするか: ファイルやフォルダを削除する操作です。" in text
    assert "なぜ確認が必要か: recursive delete" in text
    assert "判断: 内容に問題なければ「許可」。不明・危険に見える場合は「拒否」を押してください。" in text


def test_summarize_exec_approval_explains_git_fetch_in_japanese():
    text = _summarize_exec_approval("git fetch origin main --prune", "network operation")

    assert "何をするか: GitHubなどから最新情報を取得する操作です。" in text
    assert "なぜ確認が必要か: network operation" in text


def test_summarize_exec_approval_truncates_long_reason():
    text = _summarize_exec_approval("python script.py", "x" * 500)

    reason_line = [line for line in text.splitlines() if line.startswith("なぜ確認が必要か:")][0]
    assert len(reason_line) < 250
    assert reason_line.endswith("...")
