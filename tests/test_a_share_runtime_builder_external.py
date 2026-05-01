import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT_PATH_ENV = 'A_SHARE_RUNTIME_BUILDER_SCRIPT'


def load_builder_module():
    script_path = os.environ.get(SCRIPT_PATH_ENV)
    if not script_path:
        pytest.skip(f'set {SCRIPT_PATH_ENV} to run external A-share runtime builder integration tests')

    path = Path(script_path).expanduser()
    if not path.is_file():
        pytest.skip(f'{SCRIPT_PATH_ENV} does not point to a readable file: {path}')

    spec = importlib.util.spec_from_file_location('a_share_runtime_builder_external', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_postclose_brief_from_data_root_generates_minimal_brief(tmp_path):
    module = load_builder_module()
    data_root = tmp_path / 'data' / 'a-share'
    data_root.mkdir(parents=True)

    (data_root / 'market-snapshot-latest.json').write_text(
        json.dumps(
            {
                'stage': 'candidate-market-snapshot',
                'collectedAt': '2026-04-22T15:01:00Z',
                'market': {
                    'indices': [
                        {'code': '000001.SH', 'name': '上证指数', 'pct_change': 0.88},
                        {'code': '399001.SZ', 'name': '深证成指', 'pct_change': 1.21},
                    ],
                    'breadth': {
                        'universeCount': 5000,
                        'advancers': 3200,
                        'decliners': 1500,
                        'flat': 300,
                        'limitUpApprox': 85,
                        'limitDownApprox': 3,
                        'topGainers': [
                            {'code': 'sz300001', 'name': '强势股A', 'pct_change': 19.98},
                            {'code': 'sz300002', 'name': '强势股B', 'pct_change': 15.30},
                        ],
                        'topTurnover': [
                            {'code': 'sz300003', 'name': '成交核心A', 'pct_change': 9.5, 'amount': 1234567890},
                            {'code': 'sz300004', 'name': '成交核心B', 'pct_change': 7.2, 'amount': 987654321},
                        ],
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )

    (data_root / 'news-monitor-latest.json').write_text(
        json.dumps(
            {
                'stage': 'candidate-news-monitor',
                'collectedAt': '2026-04-22T15:05:00Z',
                'latestEvents': [
                    {'time': 1770000001, 'title': '题材A出现政策催化', 'source': 'cls', 'uri': 'https://example.com/a'},
                    {'time': 1770000002, 'title': '龙头股成交继续放大', 'source': 'wallstreetcn', 'uri': 'https://example.com/b'},
                ],
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )

    brief = module.build_postclose_brief_from_data_root(data_root)

    assert brief['stage'] == 'postclose'
    assert brief['coreFields']['stage'] == 'postclose'
    assert brief['coreFields']['mainTheme']
    assert brief['coreFields']['actionAdvice'] in {'跟随', '观察'}
    assert brief['strongestThemes']
    assert brief['strongestFlowStocks']
    assert brief['keyNews'][0]['title'] == '题材A出现政策催化'
    assert '收盘结论' in brief['finalConclusion']


def test_resolve_postclose_brief_meta_prefers_output_root_over_workspace(tmp_path):
    module = load_builder_module()
    workspace_root = tmp_path / 'workspace'
    output_root = tmp_path / 'output'
    (workspace_root / 'daily-briefs').mkdir(parents=True)
    (output_root / 'daily-briefs').mkdir(parents=True)

    (workspace_root / 'daily-briefs' / 'postclose-latest.json').write_text(
        json.dumps({'generatedAt': '2026-04-22T01:00:00Z', 'coreFields': {'mainTheme': '旧主线'}}, ensure_ascii=False),
        encoding='utf-8',
    )
    (output_root / 'daily-briefs' / 'postclose-latest.json').write_text(
        json.dumps({'generatedAt': '2026-04-22T02:00:00Z', 'coreFields': {'mainTheme': '新主线'}}, ensure_ascii=False),
        encoding='utf-8',
    )

    meta = module.resolve_postclose_brief_meta(workspace_root, output_root)

    assert meta['json']['coreFields']['mainTheme'] == '新主线'
    assert meta['file'].endswith('/output/daily-briefs/postclose-latest.json')


def test_build_research_validation_prefers_output_root_history_over_workspace(tmp_path):
    module = load_builder_module()
    workspace_root = tmp_path / 'workspace'
    output_root = tmp_path / 'output'
    (workspace_root / 'daily-briefs').mkdir(parents=True)
    (output_root / 'daily-briefs').mkdir(parents=True)

    def write_brief(base: Path, name: str, generated_at: str, stage: str, theme: str, advice: str):
        (base / 'daily-briefs' / name).write_text(
            json.dumps(
                {
                    'stage': stage,
                    'generatedAt': generated_at,
                    'marketTone': '测试',
                    'coreFields': {
                        'stage': stage,
                        'mainTheme': theme,
                        'actionAdvice': advice,
                    },
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )

    write_brief(workspace_root, 'postmarket-2026-04-01T07-10-57-209Z.json', '2026-04-01T07:10:57Z', 'postmarket', '旧主题', '观察')
    write_brief(workspace_root, 'premarket-2026-04-02T00-30-26-062Z.json', '2026-04-02T00:30:26Z', 'premarket', '旧主题不延续', '回避')

    write_brief(output_root, 'postmarket-2026-04-01T07-10-57-209Z.json', '2026-04-01T07:10:57Z', 'postmarket', '新主题', '跟随')
    write_brief(output_root, 'premarket-2026-04-02T00-30-26-062Z.json', '2026-04-02T00:30:26Z', 'premarket', '新主题', '跟随')
    write_brief(output_root, 'postclose-latest.json', '2026-04-22T02:00:00Z', 'postclose', 'Hermes主线', '观察')

    report = module.build_research_validation(workspace_root, output_root)

    assert report['sampleSize'] == 1
    assert report['continuationHits'] == 1
    assert report['actionAdviceAlignmentHits'] == 1
    assert report['samples'][0]['fromTheme'] == '新主题'
    assert report['samples'][0]['toTheme'] == '新主题'
    assert report['latestBriefAnchor']['mainTheme'] == 'Hermes主线'
