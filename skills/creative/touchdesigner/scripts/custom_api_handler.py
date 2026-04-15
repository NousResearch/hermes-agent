"""
Custom API Handler for TouchDesigner WebServer DAT
===================================================
Use this when mcp_webserver_base.tox fails to load its modules
(common — the .tox relies on relative paths to a modules/ folder
that often break during import).

Paste into TD Textport or run via exec(open('...').read()):
  Creates a WebServer DAT + Text DAT callback handler on port 9981.
  Implements the core endpoints the MCP server expects.

After running, test with:
  curl http://127.0.0.1:9981/api/td/server/td
"""

root = op('/project1')

# Remove broken webserver if present
old = op('/project1/mcp_webserver_base')
if old and old.valid:
    old.destroy()

# Create WebServer DAT
ws = root.create(webserverDAT, 'api_server')
ws.par.port = 9981
ws.par.active = True
ws.nodeX = -800; ws.nodeY = 500

# Create callback handler
cb = root.create(textDAT, 'api_handler')
cb.nodeX = -800; cb.nodeY = 400
cb.text = r'''
import json, traceback, io, sys

def onHTTPRequest(webServerDAT, request, response):
    uri = request.get('uri', '')
    method = request.get('method', 'GET')
    response['statusCode'] = 200
    response['statusReason'] = 'OK'
    response['headers'] = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}

    try:
        # TD sends POST body as bytes in request['data']
        raw = request.get('data', request.get('body', ''))
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        body = {}
        if raw and isinstance(raw, str) and raw.strip():
            body = json.loads(raw)
        pars = request.get('pars', {})

        if uri == '/api/td/server/td':
            response['data'] = json.dumps({
                'version': str(app.version),
                'osName': sys.platform,
                'apiVersion': '1.4.3',
                'product': 'TouchDesigner'
            })

        elif uri == '/api/td/server/exec':
            script = body.get('script', '')
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            result_val = None
            err_text = ''
            try:
                globs = {'op': op, 'ops': ops, 'me': webServerDAT, 'parent': parent,
                         'project': project, 'td': td, 'result': None,
                         'app': app, 'absTime': absTime}
                lines = script.strip().split('\n')
                if len(lines) == 1:
                    try:
                        result_val = eval(script, globs)
                    except SyntaxError:
                        exec(script, globs)
                        result_val = globs.get('result')
                else:
                    exec(script, globs)
                    result_val = globs.get('result')
            except Exception as e:
                err_text = traceback.format_exc()
            finally:
                captured = buf.getvalue()
                sys.stdout = old_stdout
            response['data'] = json.dumps({
                'result': _serialize(result_val),
                'stdout': captured,
                'stderr': err_text
            })

        elif uri == '/api/nodes':
            pp = pars.get('parentPath', ['/project1'])[0]
            p = op(pp)
            nodes = []
            if p:
                for c in p.children:
                    nodes.append({'name': c.name, 'path': c.path,
                                  'opType': c.OPType, 'family': c.family})
            response['data'] = json.dumps({'data': nodes})

        elif uri == '/api/nodes/errors':
            np = pars.get('nodePath', ['/project1'])[0]
            n = op(np)
            errors = []
            if n:
                def _collect(node, depth=0):
                    if depth > 10: return
                    e = node.errors()
                    if e:
                        errors.append({'nodePath': node.path, 'nodeName': node.name,
                                       'opType': node.OPType, 'errors': str(e)})
                    if hasattr(node, 'children'):
                        for c in node.children: _collect(c, depth+1)
                _collect(n)
            response['data'] = json.dumps({'data': errors, 'hasErrors': len(errors)>0,
                                            'errorCount': len(errors)})

        else:
            response['statusCode'] = 404
            response['data'] = json.dumps({'error': 'Unknown: ' + uri})

    except Exception as e:
        response['statusCode'] = 500
        response['data'] = json.dumps({'error': str(e), 'trace': traceback.format_exc()})

    return response

def _serialize(v):
    if v is None: return None
    if isinstance(v, (int, float, bool, str)): return v
    if isinstance(v, (list, tuple)): return [_serialize(i) for i in v]
    if isinstance(v, dict): return {str(k): _serialize(vv) for k, vv in v.items()}
    return str(v)
'''

# Point webserver to callback
ws.par.callbacks = cb.path

print("Custom API server created on port 9981")
print("Test: curl http://127.0.0.1:9981/api/td/server/td")
