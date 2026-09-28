import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from plugins.browser.browser_use import provider


def test_cloud_profile_survives_sessions_and_is_profile_scoped(tmp_path, monkeypatch):
    created, browsers = [], []
    class Cloud(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.headers['X-Browser-Use-API-Key'] == 'test-key'
            if self.path == '/profiles':
                created.append(body)
                result = {'id':f'profile-{len(created)}'}
            else:
                browsers.append(body)
                result = {'id':str(len(browsers)), 'cdpUrl':'ws://example.test/cdp'}
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    server = ThreadingHTTPServer(('127.0.0.1',0), Cloud)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setattr(provider, '_BASE_URL', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setenv('BROWSER_USE_API_KEY','test-key')
    try:
        for home in ('a','b','a'):
            monkeypatch.setenv('HERMES_HOME',str(tmp_path/home))
            assert provider.BrowserUseBrowserProvider().create_session(home)['cdp_url']
        assert len(created) == 2
        assert browsers[0]['profileId'] == browsers[2]['profileId'] != browsers[1]['profileId']
    finally:
        server.shutdown();server.server_close();thread.join()
