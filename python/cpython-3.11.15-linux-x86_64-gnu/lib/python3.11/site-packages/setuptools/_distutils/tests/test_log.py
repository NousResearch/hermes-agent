"""Tests for distutils.log"""

import logging
from distutils._log import log


class TestLog:
    def test_non_ascii(self, caplog):
        caplog.set_level(logging.DEBUG)
        log.debug('Dεbug\tMėssãge')
        log.fatal('Fαtal\tÈrrōr')
        assert caplog.messages == ['Dεbug\tMėssãge', 'Fαtal\tÈrrōr']
