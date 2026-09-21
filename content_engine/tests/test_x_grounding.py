"""Grounding retrieval fixtures are synthetic, never published evidence."""
import json
from pathlib import Path
from x_grounding import retrieve_grounding


def test_selects_related_posts_and_approved_blog_only(tmp_path):
    (tmp_path / 'approved.mdx').write_text('---\napproved: true\n---\nLocal model inference routing saves money.')
    (tmp_path / 'draft.mdx').write_text('---\napproved: false\n---\nLocal model inference secret draft.')
    posts = [{'text': 'Local inference model routing', 'url': 'https://x.com/a/status/1', 'approved': True}]
    result = retrieve_grounding('local inference routing', own_posts=posts, blog_dir=tmp_path, experience_path=tmp_path/'missing.json')
    assert len(result['own_posts']) == 1
    assert len(result['blog']) == 1
    assert result['blog'][0]['provenance']['kind'] == 'approved_blog_file'
    assert result['implementation'] == []
    assert result['coverage']['implementation'] == 'unavailable'


def test_private_unverified_and_unrelated_experience_excluded(tmp_path):
    path = tmp_path/'experiences.json'
    path.write_text(json.dumps([
        {'text':'local inference routing', 'verified':True, 'public_approved':False},
        {'text':'local inference routing', 'verified':False, 'public_approved':True},
        {'text':'football coaching weekends', 'verified':True, 'public_approved':True},
    ]))
    assert retrieve_grounding('local inference routing',own_posts=[],blog_dir=tmp_path,experience_path=path)['implementation'] == []


def test_experience_requires_receipt_and_retains_provenance(tmp_path):
    receipt = tmp_path/'receipt.txt'
    receipt.write_text('local inference routing test passed')
    import hashlib
    item = {'text':'local inference routing test passed','verified':True,'public_approved':True,
            'receipt_path':str(receipt),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    path = tmp_path/'experience.json'
    path.write_text(json.dumps([item]))
    result = retrieve_grounding('local inference routing',own_posts=[],blog_dir=tmp_path,experience_path=path)
    assert len(result['implementation']) == 1
    receipt.write_text('changed')
    assert not retrieve_grounding('local inference routing',own_posts=[],blog_dir=tmp_path,experience_path=path)['implementation']
