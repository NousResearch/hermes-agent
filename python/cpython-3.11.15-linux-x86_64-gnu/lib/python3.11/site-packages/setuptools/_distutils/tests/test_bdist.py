"""Tests for distutils.command.bdist."""

from distutils.command.bdist import bdist
from distutils.tests import support


class TestBuild(support.TempdirManager):
    def test_formats(self):
        # let's create a command and make sure
        # we can set the format
        dist = self.create_dist()[1]
        cmd = bdist(dist)
        cmd.formats = ['gztar']
        cmd.ensure_finalized()
        assert cmd.formats == ['gztar']

        # what formats does bdist offer?
        formats = [
            'bztar',
            'gztar',
            'rpm',
            'tar',
            'xztar',
            'zip',
            'ztar',
        ]
        found = sorted(cmd.format_commands)
        assert found == formats

    def test_skip_build(self):
        # bug #10946: bdist --skip-build should trickle down to subcommands
        dist = self.create_dist()[1]
        cmd = bdist(dist)
        cmd.skip_build = True
        cmd.ensure_finalized()
        dist.command_obj['bdist'] = cmd

        names = [
            'bdist_dumb',
        ]  # bdist_rpm does not support --skip-build

        for name in names:
            subcmd = cmd.get_finalized_command(name)
            if getattr(subcmd, '_unsupported', False):
                # command is not supported on this build
                continue
            assert subcmd.skip_build, f'{name} should take --skip-build from bdist'
