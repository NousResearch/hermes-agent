"""Synthetic fixtures only; no live messages or credentials."""
import json
from review_support import prepare
from operation_support import json_digest, save_new


def review_fixture(root, message, backend='gmail', account='work', text='A shop sale with 20% off.'):
    body, metadata = root/'body.json', root/'metadata.json'
    save_new(body, {'parts': [{'body': {'Text': text}, 'is_encoding_problem': False}],
                    'text_body': [0], 'html_body': [], 'attachments': []})
    save_new(metadata, message)
    prepared = prepare(body, metadata, account, backend, chunk_size=200)
    review_path, assessment_path = root/'review.json', root/'assessment.json'
    save_new(review_path, prepared)
    assessment = {'schema_version': 1, 'review_sha256': json_digest(prepared), 'provisional': False,
                  'chunk_reviews': [{'index': c['index'], 'sha256': c['sha256'],
                                     'reason': 'Synthetic sale text examined'} for c in prepared['chunks']],
                  'decision': 'REMOVE', 'pure_promotion': True, 'protected_content': False,
                  'fraud_status': 'not_suspected', 'attachment_dependent': False,
                  'rationale': 'Synthetic ordinary advertisement without an account consequence'}
    save_new(assessment_path, assessment)
    return review_path, assessment_path
