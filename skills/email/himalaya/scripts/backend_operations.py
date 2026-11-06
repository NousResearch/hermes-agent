"""Himalaya 2.1.0 adapters over shared target verification and process records.

Graph folders and Gmail labels have distinct response/command shapes. This
module verifies either target; only Graph cleanup move plans are implemented.
"""
import argparse
import copy
import json
from pathlib import Path

try:
    from .operation_support import (digest, json_digest, require_text, unique_record,
                                    run_recorded, successful_json, load_capture, save_new)
    from .graph_scan import supported_version, snapshot_hash, category_check
except ImportError:
    from operation_support import (digest, json_digest, require_text, unique_record,
                                   run_recorded, successful_json, load_capture, save_new)
    from graph_scan import supported_version, snapshot_hash, category_check

TARGETS = {'msgraph': ('folders', 'displayName', ['msgraph', 'mail-folder', 'get']),
           'gmail': ('labels', 'name', ['gmail', 'labels', 'get'])}


def base_argv(runtime, account, backend):
    if backend not in TARGETS:
        raise ValueError('No executable adapter for this backend; use its documented workflow')
    require_text(account, 'Account')
    if not supported_version(runtime.get('cli_version')):
        raise ValueError('Adapter supports the verified CLI 2.1.0 only')
    for key in ('executable', 'cwd'):
        if not Path(require_text(runtime.get(key), key)).is_absolute():
            raise ValueError('Runtime paths must be absolute native paths')
    args = [runtime['executable']]
    config = runtime.get('config', '')
    if config:
        if not Path(require_text(config, 'Config')).is_absolute():
            raise ValueError('Use an absolute config path')
        args += ['--config', config]
    return args + ['--account', account, '--backend', backend, '--json']


def target_get_argv(runtime, account, backend, target_id):
    return base_argv(runtime, account, backend) + TARGETS[backend][2] + ['--', require_text(target_id, 'Target ID')]


def target_from_capture(runtime, account, backend, selected, capture_directory):
    collection, name_key, _ = TARGETS[backend]
    invocation, result, response = successful_json(capture_directory)
    expected = target_get_argv(runtime, account, backend, selected['id'])
    if invocation.get('argv') != expected or invocation.get('cwd') != runtime['cwd'] or invocation.get('shell') is not False:
        raise ValueError('Target get invocation does not match the selected account/backend/ID/runtime')
    target = unique_record(response, collection, name_key, selected[name_key])
    if len(response[collection]) != 1 or target['id'] != selected['id']:
        raise ValueError('Retrieved target does not match selected ID and name')
    evidence = {'schema_version': 1, 'backend': backend, 'account': account,
                'runtime': copy.deepcopy(runtime), 'selected': copy.deepcopy(selected),
                'response': response, 'invocation': invocation, 'result': result,
                'capture_directory': str(Path(capture_directory)), 'target': copy.deepcopy(target)}
    evidence['sha256'] = json_digest(evidence)
    return evidence


def validate_target(evidence, account, backend, check_files=False):
    """Checks consistency, not authenticity of fabricated evidence or freshness."""
    if not isinstance(evidence, dict) or evidence.get('schema_version') != 1:
        raise ValueError('Verified destination evidence required')
    if evidence.get('account') != account or evidence.get('backend') != backend:
        raise ValueError('Destination account/backend mismatch')
    data = {k: v for k, v in evidence.items() if k != 'sha256'}
    if evidence.get('sha256') != json_digest(data):
        raise ValueError('Destination evidence changed')
    collection, name_key, _ = TARGETS[backend]
    selected = evidence['selected']
    target = unique_record(evidence['response'], collection, name_key, selected[name_key])
    invocation, result = evidence['invocation'], evidence['result']
    if (len(evidence['response'][collection]) != 1 or target['id'] != selected['id']
            or target != evidence['target']
            or invocation.get('argv') != target_get_argv(evidence['runtime'], account, backend, target['id'])
            or invocation.get('cwd') != evidence['runtime']['cwd'] or invocation.get('shell') is not False
            or result.get('status') != 'completed' or type(result.get('returncode')) is not int
            or result['returncode'] != 0):
        raise ValueError('Destination does not have a matching successful retrieval')
    if check_files:
        captured = successful_json(evidence['capture_directory'])
        if captured != (invocation, result, evidence['response']):
            raise ValueError('Saved destination capture differs from plan evidence')
    return target['id']


def verify_target(listing_path, name, account, backend, directory, cwd, executable='himalaya', config=''):
    """Read-only get using an ID selected from a supplied folder/label JSON file."""
    collection, name_key, _ = TARGETS[backend]
    listing_bytes = Path(listing_path).read_bytes()
    selected = unique_record(json.loads(listing_bytes), collection, name_key, name)
    directory = Path(directory)
    if not directory.is_absolute():
        raise ValueError('Use an absolute evidence directory')
    directory.mkdir(mode=0o700)
    save_new(directory/'request.json', {'executable': executable, 'cwd': str(cwd),
                                      'account': account, 'backend': backend, 'name': name})
    try:
        run_recorded([executable, '--version'], cwd, directory/'version')
        invocation, result, stdout = load_capture(directory/'version')
        version = stdout.decode('utf-8').strip()
        if result['status'] != 'completed' or result['returncode'] != 0 or not supported_version(version):
            raise ValueError('Unsupported or failed CLI version check')
        runtime = {'executable': invocation['argv'][0], 'cwd': invocation['cwd'],
                   'cli_version': version, 'config': config}
        args = target_get_argv(runtime, account, backend, selected['id'])
        run_recorded(args, runtime['cwd'], directory/'get')
        evidence = target_from_capture(runtime, account, backend, selected, directory/'get')
        evidence['listing_source'] = {'path': str(Path(listing_path).resolve()), 'sha256': digest(listing_bytes)}
        evidence['sha256'] = json_digest({k: v for k, v in evidence.items() if k != 'sha256'})
        save_new(directory/'target.json', evidence)
        return directory/'target.json'
    except Exception as exc:
        save_new(directory/'failure.json', {'error_type': type(exc).__name__,
                                           'note': 'Preserve this directory; use a new attempt directory'})
        raise



def graph_move_argv(runtime, account, source_id, destination_id):
    return base_argv(runtime, account, 'msgraph') + ['msgraph', 'message', 'move', '--',
                                                  require_text(source_id, 'Message ID'),
                                                  require_text(destination_id, 'Destination ID')]


def validate_graph_plan(plan, record, check_files=False):
    if plan.get('plan_schema') != 2 or record.get('backend', 'msgraph') != 'msgraph':
        raise ValueError('New execution requires a verified Graph plan schema 2')
    destination = validate_target(plan.get('destination_evidence'), record['account'], 'msgraph', check_files)
    if (plan.get('destination_folder') != destination
            or plan.get('runtime') != plan['destination_evidence']['runtime']
            or plan.get('source_id') != record['message']['id']
            or plan.get('source_folder') != record['message']['parentFolderId']
            or plan.get('source_snapshot_sha256') != snapshot_hash(record['message'])
            or plan.get('argv') != graph_move_argv(plan['runtime'], record['account'], record['message']['id'], destination)):
        raise ValueError('Plan IDs/argv/runtime do not match verified records')
    source = plan.get('source_evidence', {})
    snapshot = source.get('snapshot')
    metadata = source.get('metadata_evidence')
    if not isinstance(metadata, dict):
        raise ValueError('Source needs scanner metadata evidence')
    if (snapshot != record['message'] or source.get('account') != record['account']
            or metadata.get('scope', {}).get('config') != plan['runtime'].get('config', '')
            or metadata.get('cli_version') != plan['runtime']['cli_version']):
        raise ValueError('Source preflight does not match the record/runtime')
    # Check request/snapshot binding for any category shape, not just omission.
    try:
        from .graph_scan import _selected_list_evidence_matches
    except ImportError:
        from graph_scan import _selected_list_evidence_matches
    if not _selected_list_evidence_matches(snapshot, metadata, record['account']):
        raise ValueError('Source needs fresh scanner metadata evidence')
    if check_files:
        if plan.get('action') == 'cleanup':
            try:
                from .review_support import validate_binding
            except ImportError:
                from review_support import validate_binding
            validate_binding(plan.get('review'), record['account'], 'msgraph', plan['source_id'])
        raw = Path(source['path']).read_bytes()
        if digest(raw) != source.get('sha256'):
            raise ValueError('Source preflight file changed')
        scan = json.loads(raw)
        matches = [r for r in scan.get('messages', []) if r.get('id') == plan['source_id']]
        if (scan.get('complete') is not True or scan.get('account') != record['account']
                or scan.get('backend') != 'msgraph' or matches != [snapshot]
                or scan.get('metadata_evidence', {}).get(plan['source_id']) != metadata):
            raise ValueError('Source preflight capture does not match plan')
    return plan['argv']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=sorted(TARGETS))
    parser.add_argument('--account', required=True)
    parser.add_argument('--listing', required=True, type=Path)
    parser.add_argument('--name', required=True, help='Exact target name in a verified listing scope')
    parser.add_argument('--directory', required=True, type=Path, help='New absolute private evidence directory')
    parser.add_argument('--cwd', required=True, help='Existing absolute native working directory')
    parser.add_argument('--executable', default='himalaya')
    parser.add_argument('--config', default='')
    args = parser.parse_args()
    path = verify_target(args.listing, args.name, args.account, args.backend, args.directory,
                         args.cwd, args.executable, args.config)
    print(json.dumps({'target_evidence_file': str(path)}))


if __name__ == '__main__':
    main()
