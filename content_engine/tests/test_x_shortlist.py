"""Clean five-to-ten recommendation review contract."""
from test_x_manager_reports import _load, artifact
from test_x_lane_safety import lanes, source, draft


def test_batch_stops_at_ten_without_consuming_more_sources(lanes, monkeypatch):
    scout, _, _ = lanes
    monkeypatch.setattr(scout, '_draft', lambda row: draft())
    rows = [source(hours=1 + i / 100) for i in range(15)]
    artifacts, _, _ = scout._candidate_artifacts(rows)
    assert len(artifacts) == 10
    assert len(scout._load_verdicts()['ids']) == 10


def test_standalone_is_a_valid_source_bound_option(lanes, monkeypatch):
    scout, _, _ = lanes
    monkeypatch.setattr(scout, '_draft', lambda row: {**draft(), 'verdict': 'standalone'})
    artifacts, _, _ = scout._candidate_artifacts([source()])
    assert len(artifacts) == 1
    assert artifacts[0].lane == scout.xm.LANE_TRANSFORM
    assert artifacts[0].pack.context['sources']



def test_clean_attachment_excludes_diagnostics(tmp_path, monkeypatch, artifact):
    report = _load('x_manager_report')
    monkeypatch.setattr(report, 'REPORT_DIR', tmp_path)
    artifact.pack.context['feed_coverage'] = {'internal_trace': 'do-not-render'}
    artifact.pack.context['recommended_action'] = 'quote'
    body = report.render_report([artifact], lane='quote-scout', title='Your X shortlist', clean=True).read_text()
    assert 'Suggested post' in body and 'Quote tweet' in body
    assert 'do-not-render' not in body
    assert 'Memory hints' not in body
    assert '<details' not in body
    assert 'Full conversation unavailable' in body
    assert 'Nothing is published automatically' in body


def test_underfilled_packet_not_delivered(monkeypatch, capsys, artifact):
    scout = _load('x_quote_scout')
    monkeypatch.setattr(scout, '_load_env', lambda: None)
    monkeypatch.setattr(scout, '_collect', lambda: [{}])
    monkeypatch.setattr(scout, '_candidate_artifacts', lambda rows: ([artifact] * 4, [], []))
    monkeypatch.setattr(scout, '_merge_standalone_seeds', lambda seeds: None)
    monkeypatch.setattr(scout, '_already_reported', lambda items: False)
    monkeypatch.setattr(scout.xm, 'stage_for_approval', lambda art: art.id)
    scout.main()
    output = capsys.readouterr()
    assert not output.out
    assert '4/5' in output.err
