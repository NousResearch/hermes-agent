"""The boot-time onboarding due answer (``hermes_cli.onboarding_due``): a reader during the boot
computation waits for it, and a writer that moves the answer mid-computation is never lost."""

from __future__ import annotations

import threading

import pytest

from hermes_cli import onboarding_due


@pytest.fixture(autouse=True)
def _fresh_cache():
    onboarding_due.reset_for_tests()
    yield
    onboarding_due.reset_for_tests()


def test_a_reader_during_the_boot_computation_waits_for_it():
    finish = onboarding_due.begin_boot()
    answers: list[dict] = []
    reader = threading.Thread(target=lambda: answers.append(onboarding_due.read(lambda: {"live": True})))
    reader.start()
    finish(lambda: {"run": True, "eligible": True})
    reader.join(5)

    assert answers == [{"run": True, "eligible": True}]


def test_a_write_during_the_boot_computation_is_not_lost():
    finish = onboarding_due.begin_boot()

    def compute():
        onboarding_due.invalidate()  # set_run / a claim landed while the boot was computing
        return {"run": True, "eligible": True}
    finish(compute)

    assert onboarding_due.read(lambda: {"run": False, "eligible": True}) == {"run": False, "eligible": True}


@pytest.mark.parametrize("path", ["boot", "read"])
def test_a_key_saved_during_the_computation_is_not_hidden_by_the_cache(path):
    env = {"key": False}

    def compute():
        answer = {"run": True, "eligible": env["key"]}  # the provider check has read the files
        env["key"] = True                               # then another process saves a key
        return answer

    def stamp():
        return env["key"]

    if path == "boot":
        onboarding_due.begin_boot(stamp)(compute)
    else:
        onboarding_due.read(compute, stamp=stamp)

    assert onboarding_due.read(lambda: {"run": True, "eligible": env["key"]}, stamp=stamp)["eligible"] is True
