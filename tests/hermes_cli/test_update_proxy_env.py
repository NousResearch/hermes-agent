import os

from hermes_cli.update_proxy_env import restore_user_proxy_env


def test_marked_update_proxy_env_is_undone_and_the_users_own_env_kept():
    user = {"PATH": "/usr/bin", "no_proxy": "corp.internal", "ALL_PROXY": "socks5h://user:1080"}
    env = {**user, "HTTPS_PROXY": "http://10.0.0.7:7890", "https_proxy": "http://10.0.0.7:7890",
           "NO_PROXY": "corp.internal,localhost,10.0.0.0/8", "no_proxy": "corp.internal,localhost,10.0.0.0/8",
           "HERMES_UPDATE_SYSTEM_PROXY": "HTTPS_PROXY,https_proxy,NO_PROXY,PATH",
           "HERMES_UPDATE_USER_no_proxy": "corp.internal"}

    restore_user_proxy_env(env)

    assert env == user  # a key outside the proxy set (PATH) is never dropped
    unmarked = {"HTTPS_PROXY": "http://mine:3128", "NO_PROXY": "lan"}
    restore_user_proxy_env(unmarked)
    assert unmarked == {"HTTPS_PROXY": "http://mine:3128", "NO_PROXY": "lan"}
    assert "HERMES_UPDATE_SYSTEM_PROXY" not in os.environ
