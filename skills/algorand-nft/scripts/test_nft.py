import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from algosdk import encoding, transaction
import nft



class FakeAlgod:
    def __init__(self, genesis='testnet-v1.0', gh='SGO1GKSzyE7IEPItTxCByw9x8FmnrCDexi9/cOUJOiI=', amount=1000000, fee=0):
        self.genesis, self.gh, self.amount, self.fee = genesis, gh, amount, fee
        self.calls = []
    def suggested_params(self, **kw):
        self.calls.append(('params', kw))
        return transaction.SuggestedParams(self.fee, 100, 1100, self.gh, self.genesis, min_fee=1000)
    def account_info(self, address, **kw):
        self.calls.append(('account', kw))
        return {'amount': self.amount, 'min-balance': 100000}

class NFTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, **overrides):
        raw = b'{"name":"n","image":"ipfs://bafy/i"}\n'
        meta = self.root / 'source.json'
        meta.write_bytes(raw)
        args = dict(creator=encoding.encode_address(bytes(range(32))), name='n', unit='NFT', metadata=meta, metadata_url='ipfs://bafy/meta#arc3', out_dir=self.root/'out', algod=FakeAlgod(), fetch=lambda url: raw)
        args.update(overrides)
        return nft.prepare(**args)

    def test_metadata_deterministic_utf8(self):
        p = self.root / 'metadata.json'
        nft.write_metadata('Café', 'Description', 'ipfs://bafy/image.png', 'image/png', p)
        self.assertEqual(p.read_bytes(), b'{"description":"Description","image":"ipfs://bafy/image.png","image_mimetype":"image/png","name":"Caf\xc3\xa9"}\n')

    def test_metadata_rejects_unsafe_image(self):
        for i, image in enumerate(('http://example.com/i', 'https://user:pass@example.com/i', 'https://127.0.0.1/i', 'https://localhost/i', 'ipfs://', 'https:///image', 'https://example.com/i#bad')):
            with self.subTest(image=image), self.assertRaises(ValueError):
                nft.write_metadata('a', '', image, 'image/png', self.root / f'bad{i}.json')

    def test_metadata_validation_rejects_unsupported_documents(self):
        for obj in ([], {}, {'name': 'n'}, {'name': '', 'image': 'ipfs://bafy/i'}, {'name': 'n', 'image': 'http://example.com/i'}, {'name': 'n', 'image': 'ipfs://bafy/i', 'extra_metadata': {}}, {'name': 123, 'image': 'ipfs://bafy/i'}):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                nft.validate_metadata(json.dumps(obj).encode())
        for raw in (b'not json', b'\xff', b'{"name":"a","name":"b","image":"ipfs://bafy/i"}', b'{"name":"a","image":"ipfs://bafy/i","x":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                nft.validate_metadata(raw)
        self.assertEqual(nft.validate_metadata(b'{ "name":"n", "image":"ipfs://bafy/i" }')['name'], 'n')


    def test_prepare_roundtrip_exact_hash_immutable(self):
        creator = encoding.encode_address(bytes(range(32)))
        raw = b'{ "name": "n", "image": "ipfs://bafy/i" }\n'
        meta = self.root / 'meta.json'
        meta.write_bytes(raw)
        class FakeAlgod:
            def suggested_params(self, **kw):
                assert kw['timeout'] == 15
                return transaction.SuggestedParams(0, 100, 1100, 'SGO1GKSzyE7IEPItTxCByw9x8FmnrCDexi9/cOUJOiI=', 'testnet-v1.0', min_fee=1000)
            def account_info(self, address, **kw):
                assert address == creator
                assert kw['timeout'] == 15
                return {'amount': 1000000, 'min-balance': 100000}
        out = self.root / 'bundle'
        manifest = nft.prepare(creator, 'n', 'NFT', meta, 'ipfs://bafy/meta#arc3', out, algod=FakeAlgod(), fetch=lambda url: raw)
        txn = encoding.msgpack_decode(base64.b64encode((out / 'unsigned.txn').read_bytes()).decode())
        self.assertIsInstance(txn, transaction.AssetConfigTxn)
        self.assertEqual(txn.type, "acfg")
        self.assertEqual(txn.index, 0)
        self.assertEqual(txn.metadata_hash, hashlib.sha256(raw).digest())
        self.assertEqual((txn.total, txn.decimals, txn.default_frozen), (1, 0, False))
        for role in ('manager', 'reserve', 'freeze', 'clawback', 'rekey_to', 'group', 'lease', 'note'):
            self.assertIsNone(getattr(txn, role))
        self.assertEqual(manifest['txid'], txn.get_txid())
        self.assertEqual(manifest, json.loads((out / 'manifest.json').read_text()))
        self.assertEqual(manifest['metadata_hash']['hex'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(manifest['metadata_hash']['base64'], base64.b64encode(hashlib.sha256(raw).digest()).decode())

    def test_prepare_wrong_network_fails_closed(self):
        for client in (FakeAlgod(genesis='mainnet-v1.0'), FakeAlgod(gh='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=')):
            with self.subTest(client=client), self.assertRaisesRegex(ValueError, 'TestNet'):
                self.prepare(algod=client)
            self.assertFalse((self.root/'out').exists())

    def test_prepare_requires_available_balance_and_caps_fee(self):
        for client, message in ((FakeAlgod(amount=200999), 'balance'), (FakeAlgod(fee=10000), 'fee')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.prepare(algod=client)
            self.assertFalse((self.root/'out').exists())

    def test_prepare_rejects_remote_byte_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'match'):
            self.prepare(fetch=lambda url: b'{"name":"n","image":"ipfs://bafy/i"}')
        self.assertFalse((self.root/'out').exists())

    def test_prepare_validates_all_inputs_before_network(self):
        for inputs in (
            {'creator': 'not-an-address'}, {'name': 'é' * 17},
            {'unit': 'é' * 5}, {'metadata_url': 'https://example.com/m.json'},
            {'metadata_url': 'https://example.com/' + 'a' * 80 + '#arc3'},
            {'metadata_url': 'https://user:password@example.com/m#arc3'},
        ):
            client = FakeAlgod()
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                self.prepare(algod=client, **inputs)
            self.assertEqual(client.calls, [])
            self.assertFalse((self.root/'out').exists())

    def test_fetch_metadata_requires_public_https_exact_bytes_and_cors(self):
        self.assertTrue(callable(getattr(nft, 'fetch_metadata', None)), 'metadata fetch is not implemented')
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status = 200
        response.getheader.side_effect = lambda name: {'Access-Control-Allow-Origin': '*', 'Content-Encoding': None}.get(name)
        response.read.return_value = b'{"raw":true}\n'
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch('nft.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('93.184.216.34', 443))]), patch('nft.socket.create_connection') as connect, patch('nft.ssl.create_default_context'), patch('nft.http.client.HTTPSConnection', return_value=connection):
            self.assertEqual(nft.fetch_metadata('https://example.com/m.json#arc3'), b'{"raw":true}\n')
            connect.assert_called_with(('93.184.216.34', 443), timeout=15)
            self.assertEqual(connection.request.call_args.args[:2], ('GET', '/m.json'))
            nft.fetch_metadata('ipfs://bafy/meta#arc3')
            self.assertEqual(connection.request.call_args.args[:2], ('GET', '/ipfs/bafy/meta'))
            response.status = 302
            with self.assertRaisesRegex(ValueError, 'HTTP'):
                nft.fetch_metadata('https://example.com/m#arc3')
            response.status = 200
            response.getheader.return_value = None
            response.getheader.side_effect = None
            with self.assertRaisesRegex(ValueError, 'CORS'):
                nft.fetch_metadata('https://example.com/m#arc3')
            response.getheader.side_effect = lambda name: {'Access-Control-Allow-Origin': '*', 'Content-Encoding': None}.get(name)
            response.read.return_value = b'x' * (nft.MAX_BYTES + 1)
            with self.assertRaisesRegex(ValueError, 'MiB'):
                nft.fetch_metadata('https://example.com/m#arc3')
        with patch('nft.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 443))]), patch('nft.socket.create_connection') as connect:
            with self.assertRaisesRegex(ValueError, 'public'):
                nft.fetch_metadata('https://example.com/m#arc3')
            connect.assert_not_called()

    def test_cli_creates_metadata_without_overwriting(self):
        import subprocess, sys
        target = self.root / 'metadata.json'
        command = [sys.executable, str(Path(nft.__file__)), 'metadata', '--name', 'Demo', '--description', 'Test fixture, not minted', '--image', 'ipfs://bafy/image', '--mime', 'image/png', '--out', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(target.exists(), 'CLI did not create metadata')
        original = target.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), original)
        self.assertNotIn('Traceback', result.stderr)
        result = subprocess.run([sys.executable, str(Path(nft.__file__)), 'prepare', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--creator', result.stdout)

    def test_prepare_does_not_overwrite_existing_output(self):
        out = self.root / 'out'
        out.mkdir()
        sentinel = out / 'unsigned.txn'
        sentinel.write_bytes(b'keep me')
        client = FakeAlgod()
        with self.assertRaises(FileExistsError):
            self.prepare(algod=client)
        self.assertEqual(sentinel.read_bytes(), b'keep me')
        self.assertEqual(client.calls, [])

    def test_metadata_rejects_invalid_mime_or_description(self):
        for mime, description in (('', ''), ('not-mime', ''), ('image/png\nInjected: yes', ''), ('image/png', 123)):
            with self.subTest(mime=mime, description=description), self.assertRaises(ValueError):
                nft.write_metadata('n', description, 'ipfs://bafy/i', mime, self.root / 'bad.json')
        self.assertFalse((self.root / 'bad.json').exists())

if __name__ == '__main__':
    unittest.main()
