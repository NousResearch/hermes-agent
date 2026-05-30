from gateway.run import GatewayRunner


class TestGatewayAutoStewardPromptGuard:
    def _make_runner(self, opt_in_token="\t", armed=None):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._auto_steward_opt_in_token = opt_in_token
        runner._auto_steward_armed = armed or {}
        return runner

    def test_message_is_auto_steward_prompt_ignores_whitespace_differences(self):
        runner = self._make_runner()
        prompt = runner._build_auto_steward_prompt().replace(" ", "\n")
        assert runner._message_is_auto_steward_prompt(prompt)

    def test_should_drop_unarmed_leaked_auto_steward_prompt(self):
        runner = self._make_runner(armed={})
        assert runner._should_drop_unarmed_auto_steward_message(
            runner._build_auto_steward_prompt(),
            session_key="sess-1",
        )

    def test_should_not_drop_armed_auto_steward_prompt(self):
        runner = self._make_runner(armed={"sess-1": True})
        assert not runner._should_drop_unarmed_auto_steward_message(
            runner._build_auto_steward_prompt(),
            session_key="sess-1",
        )

    def test_should_not_drop_normal_message(self):
        runner = self._make_runner(armed={})
        assert not runner._should_drop_unarmed_auto_steward_message(
            "test",
            session_key="sess-1",
        )

    def test_empty_token_disables_guard(self):
        runner = self._make_runner(opt_in_token="", armed={})
        assert not runner._should_drop_unarmed_auto_steward_message(
            runner._build_auto_steward_prompt(),
            session_key="sess-1",
        )
