"""Linux proc-stat parsing is data-only and shared by PID identity and zombie guards."""

import pytest

from hermes_cli import _subprocess_compat as compat


def _proc_stat_row(
    comm: str, *, state: str = "S", start_time: int | str = 987654
) -> str:
    fields_4_through_21 = [str(value) for value in range(1, 19)]
    return " ".join(
        [f"123 ({comm})", state, *fields_4_through_21, str(start_time)]
    )


class TestLinuxProcStatParsing:
    @pytest.mark.parametrize(
        "comm",
        [
            "python",
            "process with spaces",
            "worker) with right paren",
            "nested (worker)) name",
        ],
    )
    def test_state_and_start_time_follow_final_comm_parenthesis(self, comm):
        raw = _proc_stat_row(comm, state="Z", start_time=424242)

        assert compat.parse_linux_proc_stat_suffix(raw)[0] == "Z"
        assert compat.parse_linux_proc_stat_start_time(raw) == 424242

    @pytest.mark.parametrize(
        "raw",
        [
            "123 python S 1 2 3",
            "123 ()",
            "123 (truncated) S 1 2",
            _proc_stat_row("bad start", start_time="not-an-integer"),
            _proc_stat_row("bad state", state="99"),
            _proc_stat_row("missing state", state=""),
        ],
    )
    def test_malformed_or_truncated_records_raise_value_error(self, raw):
        with pytest.raises(ValueError):
            compat.parse_linux_proc_stat_start_time(raw)
