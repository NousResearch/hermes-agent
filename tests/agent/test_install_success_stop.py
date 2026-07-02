from agent.install_success_stop import install_success_stop_response


def test_install_success_stop_requires_acquire_install_and_verify():
    messages = [
        {"role": "user", "content": "请帮我安装 https://github.com/BigPizzaV3/CodexPlusPlus"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"curl -L https://github.com/BigPizzaV3/CodexPlusPlus/releases/latest -o app.dmg"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_1",
            "content": '{"output":"downloaded CodexPlusPlus v1.1.7 dmg from github.com","exit_code":0}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"cp -R CodexPlusPlus.app /Applications && ls /Applications/CodexPlusPlus.app"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_2",
            "content": '{"output":"/Applications/CodexPlusPlus.app installed and verified","exit_code":0}',
        },
    ]

    response = install_success_stop_response(messages, 0)

    assert response is not None
    assert "已完成安装并通过验证" in response
    assert "v1.1.7" in response


def test_install_success_stop_does_not_fire_without_verification():
    messages = [
        {"role": "user", "content": "install https://github.com/acme/tool"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"git clone https://github.com/acme/tool && npm install"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_1",
            "content": '{"output":"downloaded and installed packages","exit_code":0}',
        },
    ]

    assert install_success_stop_response(messages, 0) is None


def test_install_success_stop_does_not_fire_on_failure():
    messages = [
        {"role": "user", "content": "install https://github.com/acme/tool"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"curl https://github.com/acme/tool && tool --version"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_1",
            "content": '{"output":"download failed: not found","exit_code":1}',
        },
    ]

    assert install_success_stop_response(messages, 0) is None


def test_install_success_stop_supports_package_manager_installs():
    messages = [
        {"role": "user", "content": "install ffmpeg with brew"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"brew install ffmpeg && ffmpeg -version"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_1",
            "content": '{"output":"ffmpeg 7.1 successfully installed\\nffmpeg version 7.1","exit_code":0}',
        },
    ]

    assert install_success_stop_response(messages, 0) is not None
