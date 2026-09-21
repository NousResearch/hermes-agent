import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'/'content_engine'))


def test_runtime_bundle_real_cron_resolver(tmp_path,monkeypatch):
    import x_runtime_bundle
    home=tmp_path/'home'
    monkeypatch.setenv('HERMES_HOME',str(home))
    x_runtime_bundle.build_bundle(home/'scripts')
    from cron.scheduler_script import _run_job_script
    ok, output=_run_job_script('x_runtime_probe.py',workdir=str(tmp_path))
    assert ok,output
    assert 'runtime imports verified' in output
    ok,output=_run_job_script('x_mix_report.py',workdir=str(tmp_path))
    assert ok,output
    report=Path(next(line[6:] for line in output.splitlines() if line.startswith('MEDIA:')))
    assert report.is_relative_to(home)
    assert 'not published posts' in report.read_text()
    assert not (home/'scripts'/'content_engine'/'db'/'content_engine.db').exists()
    ok,output=_run_job_script(str(Path(__file__).resolve()))
    assert not ok and 'outside' in output
