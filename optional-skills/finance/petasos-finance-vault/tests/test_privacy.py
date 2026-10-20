"""Privacy audit (R5.1) — no private deployment details in v2 files.

Sweeps the v2-written surfaces (petasos/, scripts/, tests/, templates/,
references/) for
the §5 classes recorded in the requirements brief: real account numbers, the
owner name, client IP, card fragments, internal model routing, delivery
channel specifics. The frozen fixture corpus is exempt BY RECORDED DECISION
(brief §4/§5) and is excluded — but these patterns must never leak into any
OTHER file.

Guard test: exists to fail the moment a private detail lands in v2 source.
Patterns are assembled from fragments so this file itself stays clean.
"""
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Assembled from fragments (never written whole in this file).
# Owner decision 14/10: the public repo carries NO personal data and NO real
# samples — fixtures/ and docs/ join the sweep (the frozen-corpus exemption is
# REVOKED; bundled fixtures are synthetic). Real personal literals for this
# deployment live ONLY in the private vault manual, never here — so this guard
# uses class sentinels: the synthetic sample data itself must stay synthetic.
ACCOUNT_NUMBERS = [
    '999999' + '999',        # tripwire (must appear nowhere)
]
OWNER_NAME = 'PRIVACY ' + 'SENTINELNAME'
CLIENT_IP = '192.0.2.' + '99'
CARD_FRAGMENTS = ['4242' + 'xx' + '4242']
MODEL_ROUTING = ['vision-' + 'capable', 'rate-' + 'limited']
DEPLOYMENT = ['to ' + 'Telegram']

SWEEP_DIRS = ['petasos', 'scripts', 'tests', 'templates', 'references',
              'fixtures', 'docs']


def iter_files():
    for base in SWEEP_DIRS:
        for dirpath, _dirs, files in os.walk(os.path.join(REPO, base)):
            for name in files:
                if name.endswith(('.py', '.md', '.yaml')):
                    yield os.path.join(dirpath, name)


def forbidden():
    """[(kind, literal, relpath, lineno)] for every hit outside fixtures/."""
    hits = []
    checks = ([('account-number', a) for a in ACCOUNT_NUMBERS]
              + [('owner-name', OWNER_NAME)]
              + [('client-ip', CLIENT_IP)]
              + [('card-fragment', c) for c in CARD_FRAGMENTS]
              + [('model-routing', m) for m in MODEL_ROUTING]
              + [('deployment', d) for d in DEPLOYMENT])
    for path in iter_files():
        rel = os.path.relpath(path, REPO)
        with open(path, encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                for kind, lit in checks:
                    if lit in line:
                        hits.append((kind, lit, rel, lineno))
    return hits


class PrivacySweep(unittest.TestCase):
    def test_no_private_details_in_v2_files(self):
        hits = forbidden()
        self.assertEqual(
            hits, [],
            'private deployment details found outside the frozen corpus:\n'
            + '\n'.join(f'{rel}:{lineno}: {kind} {lit!r}'
                        for kind, lit, rel, lineno in hits))

    def test_fixture_corpus_seal_untouched(self):
        # The corpus is sealed (SYNTHETIC samples since 14/10 — owner revoked
        # the real-capture corpus; real captures live vault-side only). Any
        # drift from these seals = someone edited samples: re-review + re-seal.
        import hashlib
        seal = {
            'acb__balance-alert__atm.txt':
                '006ab2fc5eb4c138e5f1e2e1e42db747708e87f4b443734323d430388c402b66',
            'acb__balance-alert__voucher.txt':
                'e14e7712f7a42496b18b9a5fb3bd0902083dc560f06b10a917cfc180535a4af6',
            'bidv__card-spend.txt':
                '32167092b505bbb63a9e7bc5114a4158f027d05f36b95b7853ec66fbc004d350',
            'bidv__outbound-transfer.txt':
                '5d9f6db8359baf82106b73e0cabd6c5ae45b068692870f4e6c956d5045b04deb',
            'hsbc__account-change__synthetic.txt':
                '01637db0d5ddc4e4cd02e59b7a7dbaf73e3ec3010ec34f6d285915e9a3a806ae',
            'hsbc__card-alert__synthetic.txt':
                'f00d47d7bfaa77d0ec7801a75e8f8b6d48b4509243aebed6d8e9468047e31856',
            'bidv__credit-card-payment__synthetic.txt':
                'a62c3a79fb034d3500701f4d35b060c03671563f2ba1dcd03f308f10a44d5280',
            'expected.yaml':
                '352421c19993758afda2a0d4994233ad7df4fc92e9d194f6d13bcd0d5dde86be',
        }
        for name, want in seal.items():
            with open(os.path.join(REPO, 'fixtures', name), 'rb') as f:
                got = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(got, want, f'fixtures/{name} drifted from the freeze seal')


if __name__ == '__main__':
    unittest.main()
