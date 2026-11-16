"""Prepare unsigned immutable TestNet ARC-3 NFTs. Never signs or submits."""
import ipaddress
import http.client
import socket
import ssl
import json
from pathlib import Path
from urllib.parse import urlsplit


def validate_uri(uri, arc3=False):
    if not isinstance(uri, str) or any(c.isspace() or ord(c) < 32 for c in uri) or '\\' in uri:
        raise ValueError('invalid URI')
    p = urlsplit(uri)
    if p.scheme not in ('https', 'ipfs') or not p.hostname or p.username is not None or p.password is not None:
        raise ValueError('URI must use https:// or ipfs:// without credentials')
    if (arc3 and (p.fragment != 'arc3' or uri.count('#') != 1)) or (not arc3 and '#' in uri):
        raise ValueError('metadata URL must end #arc3' if arc3 else 'image URI cannot contain a fragment')
    try:
        port = p.port
    except ValueError:
        raise ValueError('invalid URI port') from None
    host = p.hostname.lower().rstrip('.')
    if host == 'localhost' or host.endswith(('.localhost', '.local')):
        raise ValueError('local hosts are not allowed')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError('private addresses are not allowed')
    if p.scheme == 'ipfs' and (port is not None or p.query):
        raise ValueError('invalid IPFS URI')
    return p


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError('nonstandard JSON number')


def validate_metadata(raw):
    try:
        obj = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError('metadata must be a UTF-8 JSON object') from e
    if not isinstance(obj, dict) or 'extra_metadata' in obj:
        raise ValueError('metadata must be an object without extra_metadata')
    if not isinstance(obj.get('name'), str) or not obj['name'].strip():
        raise ValueError('metadata requires a nonempty name')
    validate_uri(obj.get('image'))
    if 'description' in obj and not isinstance(obj['description'], str):
        raise ValueError('description must be a string')
    if 'image_mimetype' in obj:
        import re
        mime = obj['image_mimetype']
        if not isinstance(mime, str) or not re.fullmatch(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+', mime):
            raise ValueError('image_mimetype must be a MIME type such as image/png')
    return obj


def write_metadata(name, description, image, mime, out):
    data = dict(name=name, description=description, image=image, image_mimetype=mime)
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    validate_metadata(raw)
    with Path(out).open('xb') as f:
        f.write(raw)

# TestNet identity corroborated against live /v2/transactions/params.
# Network genesis source: algorand/go-algorand installer/genesis/testnet/genesis.json.
TESTNET_ID = 'testnet-v1.0'
TESTNET_HASH = 'SGO1GKSzyE7IEPItTxCByw9x8FmnrCDexi9/cOUJOiI='
DEFAULT_ALGOD = 'https://testnet-api.algonode.cloud'
TIMEOUT = 15
MAX_BYTES = 1048576


def fetch_metadata(uri):
    """Fetch bounded identity bytes over verified TLS; no proxies or redirects.

    Pin a DNS-resolved public IP to the connection, retaining hostname-based
    certificate verification/SNI to avoid DNS-rebinding and private-network SSRF.
    """
    p = validate_uri(uri, arc3=True)
    target = uri.rsplit('#', 1)[0]
    if p.scheme == 'ipfs':
        target = 'https://ipfs.io/ipfs/' + p.netloc + p.path
    p = urlsplit(target)
    port = p.port or 443
    resolved = socket.getaddrinfo(p.hostname, port, type=socket.SOCK_STREAM)
    addresses = [entry[4][0] for entry in resolved]
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError('metadata host must resolve only to public IP addresses')
    context = ssl.create_default_context()
    conn = http.client.HTTPSConnection(p.hostname, port, timeout=TIMEOUT, context=context)
    plain = socket.create_connection((addresses[0], port), timeout=TIMEOUT)
    try:
        conn.sock = context.wrap_socket(plain, server_hostname=p.hostname)
        path = p.path or '/'
        if p.query:
            path += '?' + p.query
        conn.request('GET', path, headers={'Accept': 'application/json', 'Accept-Encoding': 'identity', 'User-Agent': 'Hermes-Algorand-NFT/0.1'})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f'metadata HTTP status {response.status}; redirects are not followed')
        if response.getheader('Access-Control-Allow-Origin') != '*':
            raise ValueError('metadata endpoint must return CORS Access-Control-Allow-Origin: *')
        if response.getheader('Content-Encoding') not in (None, '', 'identity'):
            raise ValueError('metadata endpoint must return uncompressed bytes')
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('metadata exceeds 1 MiB')
        return raw
    finally:
        conn.close()
        plain.close()


def prepare(creator, name, unit, metadata, metadata_url, out_dir, *, algod_url=DEFAULT_ALGOD, algod=None, fetch=None):
    import base64
    import hashlib
    from algosdk import encoding, transaction
    from algosdk.v2client.algod import AlgodClient
    if not encoding.is_valid_address(creator):
        raise ValueError('invalid creator address or checksum')
    for label, value, limit in (('name', name, 32), ('unit', unit, 8), ('metadata URL', metadata_url, 96)):
        if not isinstance(value, str) or not value.strip() or len(value.encode('utf-8')) > limit:
            raise ValueError(f'{label} must contain 1..{limit} UTF-8 bytes')
    validate_uri(metadata_url, arc3=True)
    validate_uri(algod_url)
    if urlsplit(algod_url).scheme != 'https':
        raise ValueError('algod endpoint must use HTTPS')
    if Path(out_dir).exists():
        raise FileExistsError('output directory already exists')
    with Path(metadata).open('rb') as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('metadata exceeds 1 MiB')
    validate_metadata(raw)
    client = algod if algod is not None else AlgodClient('', algod_url)
    sp = client.suggested_params(timeout=TIMEOUT)
    if sp.gen != TESTNET_ID or sp.gh != TESTNET_HASH:
        raise ValueError('only pinned TestNet genesis is supported')
    account = client.account_info(creator, timeout=TIMEOUT)
    remote = (fetch or fetch_metadata)(metadata_url)
    if remote != raw:
        raise ValueError('remote metadata bytes do not match local file')
    digest = hashlib.sha256(raw).digest()
    txn = transaction.AssetCreateTxn(creator, sp, 1, 0, False, manager=None, reserve=None, freeze=None, clawback=None, unit_name=unit, asset_name=name, url=metadata_url, metadata_hash=digest)
    if not isinstance(txn.fee, int) or txn.fee < 1000 or txn.fee > 10000:
        raise ValueError('fee outside permitted 1000..10000 microAlgos')
    if account['amount'] - account['min-balance'] < txn.fee + 100000:
        raise ValueError('insufficient spendable balance for fee and asset minimum balance')
    manifest = {
        'network': 'testnet', 'genesis_id': sp.gen, 'genesis_hash': sp.gh,
        'creator': creator, 'roles': {k: None for k in ('manager', 'reserve', 'freeze', 'clawback')},
        'policy': 'Irrevocable: no manager, reserve, freeze or clawback; cannot reconfigure or destroy.',
        'fee': txn.fee, 'first_round': txn.first_valid_round, 'last_round': txn.last_valid_round,
        'metadata_hash': {'hex': digest.hex(), 'base64': base64.b64encode(digest).decode('ascii')},
        'txid': txn.get_txid(), 'asset': {'name': name, 'unit': unit, 'total': 1, 'decimals': 0, 'default_frozen': False, 'url': metadata_url},
        'metadata_path': str(Path(metadata).resolve()), 'metadata_url': metadata_url,
    }
    out = Path(out_dir)
    out.mkdir()
    with (out / 'unsigned.txn').open('xb') as f:
        f.write(base64.b64decode(encoding.msgpack_encode(txn)))
    with (out / 'manifest.json').open('x', encoding='utf-8') as f:
        f.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    return manifest


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    metadata = commands.add_parser('metadata', help='write ARC-3 JSON; does not upload')
    for flag in ('name', 'description', 'image', 'mime', 'out'):
        metadata.add_argument('--' + flag, required=True)
    mint = commands.add_parser('prepare', help='prepare unsigned TestNet asset creation; never submits')
    for flag in ('creator', 'name', 'unit', 'metadata', 'metadata-url', 'out-dir'):
        mint.add_argument('--' + flag, required=True)
    mint.add_argument('--algod-url', default=DEFAULT_ALGOD)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command == 'metadata':
            write_metadata(**args)
            print(json.dumps({'status': 'metadata-created-not-published', 'path': str(Path(args['out']).resolve())}))
        else:
            manifest = prepare(**args)
            print(json.dumps({'status': 'prepared-unsigned-not-submitted', 'directory': str(Path(args['out_dir']).resolve()), 'txid': manifest['txid'], 'fee': manifest['fee']}))
        return 0
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
