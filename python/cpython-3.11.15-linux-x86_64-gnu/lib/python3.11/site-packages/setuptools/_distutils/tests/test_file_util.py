"""Tests for distutils.file_util."""

import errno
import os
import unittest.mock as mock
from distutils.errors import DistutilsFileError
from distutils.file_util import copy_file, move_file

import jaraco.path
import pytest


@pytest.fixture(autouse=True)
def stuff(request, tmp_path):
    self = request.instance
    self.source = tmp_path / 'f1'
    self.target = tmp_path / 'f2'
    self.target_dir = tmp_path / 'd1'


class TestFileUtil:
    def test_move_file_verbosity(self, caplog):
        jaraco.path.build({self.source: 'some content'})

        move_file(self.source, self.target, verbose=False)
        assert not caplog.messages

        # back to original state
        move_file(self.target, self.source, verbose=False)

        move_file(self.source, self.target, verbose=True)
        wanted = [f'moving {self.source} -> {self.target}']
        assert caplog.messages == wanted

        # back to original state
        move_file(self.target, self.source, verbose=False)

        caplog.clear()
        # now the target is a dir
        os.mkdir(self.target_dir)
        move_file(self.source, self.target_dir, verbose=True)
        wanted = [f'moving {self.source} -> {self.target_dir}']
        assert caplog.messages == wanted

    def test_move_file_exception_unpacking_rename(self):
        # see issue 22182
        with (
            mock.patch("os.rename", side_effect=OSError("wrong", 1)),
            pytest.raises(DistutilsFileError),
        ):
            jaraco.path.build({self.source: 'spam eggs'})
            move_file(self.source, self.target, verbose=False)

    def test_move_file_exception_unpacking_unlink(self):
        # see issue 22182
        with (
            mock.patch("os.rename", side_effect=OSError(errno.EXDEV, "wrong")),
            mock.patch("os.unlink", side_effect=OSError("wrong", 1)),
            pytest.raises(DistutilsFileError),
        ):
            jaraco.path.build({self.source: 'spam eggs'})
            move_file(self.source, self.target, verbose=False)

    def test_copy_file_hard_link(self):
        jaraco.path.build({self.source: 'some content'})
        # Check first that copy_file() will not fall back on copying the file
        # instead of creating the hard link.
        try:
            os.link(self.source, self.target)
        except OSError as e:
            self.skipTest(f'os.link: {e}')
        else:
            self.target.unlink()
        st = os.stat(self.source)
        copy_file(self.source, self.target, link='hard')
        st2 = os.stat(self.source)
        st3 = os.stat(self.target)
        assert os.path.samestat(st, st2), (st, st2)
        assert os.path.samestat(st2, st3), (st2, st3)
        assert self.source.read_text(encoding='utf-8') == 'some content'

    def test_copy_file_hard_link_failure(self):
        # If hard linking fails, copy_file() falls back on copying file
        # (some special filesystems don't support hard linking even under
        #  Unix, see issue #8876).
        jaraco.path.build({self.source: 'some content'})
        st = os.stat(self.source)
        with mock.patch("os.link", side_effect=OSError(0, "linking unsupported")):
            copy_file(self.source, self.target, link='hard')
        st2 = os.stat(self.source)
        st3 = os.stat(self.target)
        assert os.path.samestat(st, st2), (st, st2)
        assert not os.path.samestat(st2, st3), (st2, st3)
        for fn in (self.source, self.target):
            assert fn.read_text(encoding='utf-8') == 'some content'
