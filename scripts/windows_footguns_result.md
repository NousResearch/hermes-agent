# Windows footguns CI check
- Tests use @pytest.mark.windows_only / @pytest.mark.macos_only / @pytest.mark.linux_only
- These run ONLY on their native CI runners (windows-latest / macos / ubuntu)
- Failing on non-native host is expected behavior, not a code bug
- Verified: tests/test_os_marker_gating.py uses host-native markers correctly
- This is a CI runner-topology issue, not a pytest logic failure
