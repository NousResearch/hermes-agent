"""Offline, backend-neutral shared-read JSON decoding and review evidence.

Chunks must actually be read. Hashes/coverage reject mismatches, not dishonest
claims or mistaken semantic judgments. No network, classification or authority.
"""
import argparse
import json
from pathlib import Path
try:
    from .operation_support import digest, json_digest, require_text, save_new
    from .mime_inspect import TextAndLinks
except ImportError:
    from operation_support import digest, json_digest, require_text, save_new
    from mime_inspect import TextAndLinks


def decode_shared(payload):
    if not isinstance(payload, dict) or 'error' in payload or not isinstance(payload.get('parts'), list):
        raise ValueError('Expected shared message-read JSON parts')
    parts = payload['parts']
    def indices(key):
        value = payload.get(key)
        if not isinstance(value, list) or any(type(i) is not int or not 0 <= i < len(parts) for i in value):
            raise ValueError('Missing/invalid shared body index array: '+key)
        return value
    attachments = set(indices('attachments'))
    selected = sorted(set(indices('text_body')+indices('html_body'))-attachments)
    warnings, blocks = [], []
    # Inspect all designated alternatives: a plain alternative can contain only
    # boilerplate while HTML carries the actual message. Never scan attachments.
    for i in selected:
        part = parts[i]
        if not isinstance(part, dict) or not isinstance(part.get('body'), dict):
            raise ValueError('Selected body part has no supported content')
        if part.get('is_encoding_problem') is True:
            warnings.append('encoding_problem:'+str(i))
        body = part['body']
        found = False
        for kind in ('Text', 'Html'):
            if kind not in body:
                continue
            found = True
            text = body[kind]
            if not isinstance(text, str):
                raise ValueError('Body content is not text')
            if '\ufffd' in text:
                warnings.append('replacement_character:'+str(i))
            if kind == 'Html':
                parser = TextAndLinks()
                parser.feed(text)
                parser.close()
                text = ''.join(parser.text).strip()
            if text.strip():
                blocks.append({'part': i, 'kind': kind, 'text': text})
        if not found:
            raise ValueError('Unsupported selected body content; use verified raw MIME workflow')
    if not blocks:
        warnings.append('no_readable_body')
    return {'blocks': blocks, 'warnings': warnings, 'attachment_count': len(attachments)}


def prepare(body_path, metadata_path, account, backend, chunk_size=4000):
    if type(chunk_size) is not int or not 200 <= chunk_size <= 8000:
        raise ValueError('Chunk size must be 200..8000 characters')
    body_path, metadata_path = Path(body_path).resolve(), Path(metadata_path).resolve()
    raw, metadata_raw = body_path.read_bytes(), metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    mid = require_text(metadata.get('id'), 'Metadata message ID')
    decoded = decode_shared(json.loads(raw))
    chunks = []
    for block in decoded['blocks']:
        for offset in range(0, len(block['text']), chunk_size):
            text = block['text'][offset:offset+chunk_size]
            chunks.append({'index': len(chunks), 'part': block['part'], 'kind': block['kind'],
                           'offset': offset, 'text': text, 'sha256': digest(text.encode())})
    return {'schema_version': 1, 'account': require_text(account, 'Account'),
            'backend': require_text(backend, 'Backend'), 'message_id': mid,
            'body_path': str(body_path), 'body_sha256': digest(raw),
            'metadata_path': str(metadata_path), 'metadata_sha256': digest(metadata_raw),
            'chunk_size': chunk_size, 'chunks': chunks,
            'warnings': decoded['warnings'], 'attachment_count': decoded['attachment_count'],
            'note': 'Prepared text is not reviewed. Read every chunk; HTML is text extraction, not visual rendering.'}


def load_review(path):
    review = json.loads(Path(path).read_bytes())
    fresh = prepare(review['body_path'], review['metadata_path'], review['account'],
                    review['backend'], review['chunk_size'])
    if fresh != review:
        raise ValueError('Review/source evidence changed; prepare and review current source')
    return review


def validate(review, assessment):
    if review.get('warnings') or not review.get('chunks'):
        raise ValueError('Body decoding incomplete; resolve warnings before removal')
    if (assessment.get('schema_version') != 1
            or assessment.get('review_sha256') != json_digest(review)
            or assessment.get('provisional') is not False):
        raise ValueError('Need a current non-provisional review assessment')
    rows = assessment.get('chunk_reviews')
    if not isinstance(rows, list) or len(rows) != len(review['chunks']):
        raise ValueError('Every complete body chunk must be reviewed')
    for expected, actual in zip(review['chunks'], rows):
        if (actual.get('index') != expected['index'] or actual.get('sha256') != expected['sha256']
                or not isinstance(actual.get('reason'), str) or not actual['reason'].strip()):
            raise ValueError('Chunk review is missing, reordered or changed')
    if (assessment.get('decision') != 'REMOVE' or assessment.get('pure_promotion') is not True
            or assessment.get('protected_content') is not False
            or assessment.get('fraud_status') != 'not_suspected'
            or assessment.get('attachment_dependent') is not False
            or not isinstance(assessment.get('rationale'), str) or not assessment['rationale'].strip()):
        raise ValueError('No supported pure-promotion removal assessment')
    return True


def bind(review_path, assessment_path, account, backend, message_id):
    review = load_review(review_path)
    assessment = json.loads(Path(assessment_path).read_bytes())
    if (review['account'], review['backend'], review['message_id']) != (account, backend, message_id):
        raise ValueError('Review account/backend/message mismatch')
    validate(review, assessment)
    return {'review_path': str(Path(review_path).resolve()), 'review_sha256': json_digest(review),
            'assessment_path': str(Path(assessment_path).resolve()),
            'assessment_sha256': json_digest(assessment)}


def validate_binding(binding, account, backend, message_id):
    if not isinstance(binding, dict) or bind(binding['review_path'], binding['assessment_path'],
                                           account, backend, message_id) != binding:
        raise ValueError('Plan review evidence changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--account')
    parser.add_argument('--backend')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--show', type=Path, help='Display one complete prepared chunk')
    parser.add_argument('--chunk', type=int)
    parser.add_argument('--assessment-template', type=Path, help='Write an incomplete template; not a passing decision')
    args = parser.parse_args()
    if args.show is not None:
        review = load_review(args.show)
        if args.chunk is None or not 0 <= args.chunk < len(review['chunks']):
            parser.error('Supply --chunk with an index from the prepared review')
        chunk = review['chunks'][args.chunk]
        print('BEGIN CHUNK', chunk['index'], 'CHARS', len(chunk['text']), 'SHA256', chunk['sha256'])
        print(chunk['text'])
        print('END CHUNK', chunk['index'], 'SHA256', chunk['sha256'])
        return
    if any(getattr(args, k) is None for k in ('body', 'metadata', 'account', 'backend', 'output')):
        parser.error('Preparing requires --body --metadata --account --backend --output')
    review = prepare(args.body, args.metadata, args.account, args.backend)
    save_new(args.output, review)
    if args.assessment_template:
        save_new(args.assessment_template, {
            'schema_version': 1, 'review_sha256': json_digest(review), 'provisional': True,
            'chunk_reviews': [{'index': c['index'], 'sha256': c['sha256'], 'reason': ''}
                              for c in review['chunks']],
            'decision': 'REVIEW', 'pure_promotion': None, 'protected_content': None,
            'fraud_status': 'unknown', 'attachment_dependent': None, 'rationale': ''})
    print(json.dumps({'chunks': len(review['chunks']), 'warnings': review['warnings'],
                      'review_sha256': json_digest(review), 'status': 'prepared_not_reviewed'}))


if __name__ == '__main__':
    main()
