import io
import tarfile

import pytest

from setuptools import archive_util


@pytest.fixture
def tarfile_with_unicode(tmpdir):
    """
    Create a tarfile containing only a file whose name is
    a zero byte file called testimäge.png.
    """
    tarobj = io.BytesIO()

    with tarfile.open(fileobj=tarobj, mode="w:gz") as tgz:
        data = b""

        filename = "testimäge.png"

        t = tarfile.TarInfo(filename)
        t.size = len(data)

        tgz.addfile(t, io.BytesIO(data))

    target = tmpdir / 'unicode-pkg-1.0.tar.gz'
    with open(str(target), mode='wb') as tf:
        tf.write(tarobj.getvalue())
    return str(target)


@pytest.mark.xfail(reason="#710 and #712")
def test_unicode_files(tarfile_with_unicode, tmpdir):
    target = tmpdir / 'out'
    archive_util.unpack_archive(tarfile_with_unicode, str(target))
