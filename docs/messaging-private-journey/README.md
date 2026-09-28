# Find a private Group Chat, send once, and return to its details

This bounded #98073 consumer lets an authorized private messaging recipient use the stable reference shown by `/group list`, send text with `/group N send ...`, and follow the acknowledgement back to `/group N`. Details expose the consented room's status, Bots and bounded recent messages. A native numeric reference never falls through to an unrelated legacy room.

Inventory, room-read and Send consent are distinct. Revocation or replacement invalidates the original request; a same-message replay uses the canonical receipt rather than admitting another task. A transport failure after possible commit is uncertainty, not permission to resend.

## Ownership and landing prerequisites

The six consumer source/test paths belong to #98073. No lower implementation or lower commits are added to its review delta. The independent prerequisites are:

- Permission #111939: `34332b47b3fb3c8394879e7179e157b89be115de` — private recipient identity, consent and canonical read/Send dispatch; also the reconstruction substrate.
- Route #100016: `1fa3c0addd0c3eec671f3019c443dd3e449db134` — hosted service, state/status, route and attachment contracts consumed by that dispatch. `ROUTE` in `compose.py` identifies the exact public blobs.
- Selected-route lifetime #116477: `8c36f8dd3130397c6388d7517d51a0ed8c700279` — the published SessionDB external-writer guard needed by Send through commit/rollback.
- Earlier Messaging #98073: `73fbc700664c56eabd9d7a55f178320662ef0c47` — existing legacy helpers retained by the bridge.

These dependencies do not acquire an upper-consumer prerequisite. The recipe fetches immutable public commits directly, copies named public modules, and performs only two mechanical shared-file joins: append the Permission private-admin policy definitions to the existing Messaging policy, and retain the earlier Messaging slash-access helper definitions alongside the Permission surface. It does not invent authority, replace denied controls, or execute a second runtime.

## Reproduce

From a checkout of this published Messaging revision, choose a new destination under an existing parent:

```sh
CANDIDATE=$(git rev-parse HEAD)
python3 docs/messaging-private-journey/compose.py "$PWD/../messaging-replay" --candidate-commit "$CANDIDATE"
cd ../messaging-replay
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
scripts/run_tests.sh -j 1 --file-retries 0 \
  tests/gateway/test_canonical_group_private_journey.py \
  tests/gateway/test_canonical_group_messaging_send.py \
  tests/gateway/test_canonical_group_private_read.py
```

Git, network access to the declared public repository and the project's Python development dependencies are required. `RECONSTRUCTION.json` records the pins, joins and six consumer digests. Destination reuse is rejected. `--candidate-dir` is a development-only alternative that checks the frozen six-file digest manifest; it is not public-source acceptance.

For a restricted test host, use synthetic HOME and TEMP/TMP under its permitted workspace. The canonical runner forwards HOME/TEMP/TMP; an external interpreter wrapper may serialize its compile preparation and direct Python caches into that workspace. No test-runner adaptation belongs in the product diff.

## Verification and limits

The parent reconstructed fresh public lower inputs with the frozen consumer and ran the canonical three-file gate: **46 passed, 0 failed**. The registered journey covers idle, runner-busy and adapter-busy dispatch. The source review found no reachable privacy/correctness blocker in the six-path consumer. Earlier reconstruction failures were retained and resolved by including already-published lower owners, not by changing the consumer or weakening a check.

This is a focused synthetic command/authority journey. It is not hosted-CI approval, live platform delivery, native Files/Save acceptance, cross-host recovery, all-profile support, or completion of #97681. Native Stop, approvals and Files commands are not added by this increment; unsupported numeric commands are refused rather than routed into the legacy namespace. Existing legacy behavior outside this canonical private route remains separately scoped. The programme's fixed runtime comparison is not advanced by this recipe; its execution substrate is explicitly the Permission pin above.

Original contributor attribution and earlier source history remain in the existing PR and Git history. This increment was implemented and reviewed with AI assistance.
