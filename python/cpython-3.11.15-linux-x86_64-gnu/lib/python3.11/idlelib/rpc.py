"""RPC Implementation, originally written for the Python Idle IDE

For security reasons, GvR requested that Idle's Python execution server process
connect to the Idle process, which listens for the connection.  Since Idle has
only one client per server, this was not a limitation.

   +---------------------------------+ +-------------+
   | socketserver.BaseRequestHandler | | SocketIO    |
   +---------------------------------+ +-------------+
                   ^                   | register()  |
                   |                   | unregister()|
                   |                   +-------------+
                   |                      ^  ^
                   |                      |  |
                   | + -------------------+  |
                   | |                       |
   +-------------------------+        +-----------------+
   | RPCHandler              |        | RPCClient       |
   | [attribute of RPCServer]|        |                 |
   +-------------------------+        +-----------------+

The RPCServer handler class is expected to provide register/unregister methods.
RPCHandler inherits the mix-in class SocketIO, which provides these methods.

See the Idle run.main() docstring for further information on how this was
accomplished in Idle.

"""
import builtins
import copyreg
import io
import marshal
import os
import pickle
import queue
import select
import socket
import socketserver
import struct
import sys
import threading
import traceback
import types

def unpickle_code(ms):
    "Return code object from marshal string ms."
    co = marshal.loads(ms)
    assert isinstance(co, types.CodeType)
    return co

def pickle_code(co):
    "Return unpickle function and tuple with marshalled co code object."
    assert isinstance(co, types.CodeType)
    ms = marshal.dumps(co)
    return unpickle_code, (ms,)

def dumps(obj, protocol=None):
    "Return pickled (or marshalled) string for obj."
    # IDLE passes 'None' to select pickle.DEFAULT_PROTOCOL.
    f = io.BytesIO()
    p = CodePickler(f, protocol)
    p.dump(obj)
    return f.getvalue()


class CodePickler(pickle.Pickler):
    dispatch_table = {types.CodeType: pickle_code, **copyreg.dispatch_table}


BUFSIZE = 8*1024
LOCALHOST = '127.0.0.1'

class RPCServer(socketserver.TCPServer):

    def __init__(self, addr, handlerclass=None):
        if handlerclass is None:
            handlerclass = RPCHandler
        socketserver.TCPServer.__init__(self, addr, handlerclass)

    def server_bind(self):
        "Override TCPServer method, no bind() phase for connecting entity"
        pass

    def server_activate(self):
        """Override TCPServer method, connect() instead of listen()

        Due to the reversed connection, self.server_address is actually the
        address of the Idle Client to which we are connecting.

        """
        self.socket.connect(self.server_address)

    def get_request(self):
        "Override TCPServer method, return already connected socket"
        return self.socket, self.server_address

    def handle_error(self, request, client_address):
        """Override TCPServer method

        Error message goes to __stderr__.  No error message if exiting
        normally or socket raised EOF.  Other exceptions not handled in
        server code will cause os._exit.

        """
        try:
            raise
        except SystemExit:
            raise
        except:
            erf = sys.__stderr__
            print('\n' + '-'*40, file=erf)
            print('Unhandled server exception!', file=erf)
            print('Thread: %s' % threading.current_thread().name, file=erf)
            print('Client Address: ', client_address, file=erf)
            print('Request: ', repr(request), file=erf)
            traceback.print_exc(file=erf)
            print('\n*** Unrecoverable, server exiting!', file=erf)
            print('-'*40, file=erf)
            os._exit(0)

#----------------- end class RPCServer --------------------

objecttable = {}
request_queue = queue.Queue(0)
response_queue = queue.Queue(0)


class SocketIO:

    nextseq = 0

    def __init__(self, sock, objtable=None, debugging=None):
        self.sockthread = threading.current_thread()
        if debugging is not None:
            self.debugging = debugging
        self.sock = sock
        if objtable is None:
            objtable = objecttable
        self.objtable = objtable
        self.responses = {}
        self.cvars = {}

    def close(self):
        sock = self.sock
        self.sock = None
        if sock is not None:
            sock.close()

    def exithook(self):
        "override for specific exit action"
        os._exit(0)

    def debug(self, *args):
        if not self.debugging:
            return
        s = self.location + " " + str(threading.current_thread().name)
        for a in args:
            s = s + " " + str(a)
        print(s, file=sys.__stderr__)

    def register(self, oid, object_):
        self.objtable[oid] = object_

    def unregister(self, oid):
        try:
            del self.objtable[oid]
        except KeyError:
            pass

    def localcall(self, seq, request):
        self.debug("localcall:", request)
        try:
            how, (oid, methodname, args, kwargs) = request
        except TypeError:
            return ("ERROR", "Bad request format")
        if oid not in self.objtable:
            return ("ERROR", f"Unknown object id: {oid!r}")
        obj = self.objtable[oid]
        if methodname == "__methods__":
            methods = {}
            _getmethods(obj, methods)
            return ("OK", methods)
        if methodname == "__attributes__":
            attributes = {}
            _getattributes(obj, attributes)
            return ("OK", attributes)
        if not hasattr(obj, methodname):
            return ("ERROR", f"Unsupported method name: {methodname!r}")
        method = getattr(obj, methodname)
        try:
            if how == 'CALL':
                ret = method(*args, **kwargs)
                if isinstance(ret, RemoteObject):
                    ret = remoteref(ret)
                return ("OK", ret)
            elif how == 'QUEUE':
                request_queue.put((seq, (method, args, kwargs)))
                return("QUEUED", None)
            else:
                return ("ERROR", "Unsupported message type: %s" % how)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            raise
        except OSError:
            raise
        except Exception as ex:
            return ("CALLEXC", ex)
        except:
            msg = "*** Internal Error: rpc.py:SocketIO.localcall()\n\n"\
                  " Object: %s \n Method: %s \n Args: %s\n"
            print(msg % (oid, method, args), file=sys.__stderr__)
            traceback.print_exc(file=sys.__stderr__)
            return ("EXCEPTION", None)

    def remotecall(self, oid, methodname, args, kwargs):
        self.debug("remotecall:asynccall: ", oid, methodname)
        seq = self.asynccall(oid, methodname, args, kwargs)
        return self.asyncreturn(seq)

    def remotequeue(self, oid, methodname, args, kwargs):
        self.debug("remotequeue:asyncqueue: ", oid, methodname)
        seq = self.asyncqueue(oid, methodname, args, kwargs)
        return self.asyncreturn(seq)

    def asynccall(self, oid, methodname, args, kwargs):
        request = ("CALL", (oid, methodname, args, kwargs))
        seq = self.newseq()
        if threading.current_thread() != self.sockthread:
            cvar = threading.Condition()
            self.cvars[seq] = cvar
        self.debug(("asynccall:%d:" % seq), oid, methodname, args, kwargs)
        self.putmessage((seq, request))
        return seq

    def asyncqueue(self, oid, methodname, args, kwargs):
        request = ("QUEUE", (oid, methodname, args, kwargs))
        seq = self.newseq()
        if threading.current_thread() != self.sockthread:
            cvar = threading.Condition()
            self.cvars[seq] = cvar
        self.debug(("asyncqueue:%d:" % seq), oid, methodname, args, kwargs)
        self.putmessage((seq, request))
        return seq

    def asyncreturn(self, seq):
        self.debug("asyncreturn:%d:call getresponse(): " % seq)
        response = self.getresponse(seq, wait=0.05)
        self.debug(("asyncreturn:%d:response: " % seq), response)
        return self.decoderesponse(response)

    def decoderesponse(self, response):
        how, what = response
        if how == "OK":
            return what
        if how == "QUEUED":
            return None
        if how == "EXCEPTION":
            self.debug("decoderesponse: EXCEPTION")
            return None
        if how == "EOF":
            self.debug("decoderesponse: EOF")
            self.decode_interrupthook()
            return None
        if how == "ERROR":
            self.debug("decoderesponse: Internal ERROR:", what)
            raise RuntimeError(what)
        if how == "CALLEXC":
            self.debug("decoderesponse: Call Exception:", what)
            raise what
        raise SystemError(how, what)

    def decode_interrupthook(self):
        ""
        raise EOFError

    def mainloop(self):
        """Listen on socket until I/O not ready or EOF

        pollresponse() will loop looking for seq number None, which
        never comes, and exit on EOFError.

        """
        try:
            self.getresponse(myseq=None, wait=0.05)
        except EOFError:
            self.debug("mainloop:return")
            return

    def getresponse(self, myseq, wait):
        response = self._getresponse(myseq, wait)
        if response is not None:
            how, what = response
            if how == "OK":
                response = how, self._proxify(what)
        return response

    def _proxify(self, obj):
        if isinstance(obj, RemoteProxy):
            return RPCProxy(self, obj.oid)
        if isinstance(obj, list):
            return list(map(self._proxify, obj))
        # XXX Check for other types -- not currently needed
        return obj

    def _getresponse(self, myseq, wait):
        self.debug("_getresponse:myseq:", myseq)
        if threading.current_thread() is self.sockthread:
            # this thread does all reading of requests or responses
            while True:
                response = self.pollresponse(myseq, wait)
                if response is not None:
                    return response
        else:
            # wait for notification from socket handling thread
            cvar = self.cvars[myseq]
            cvar.acquire()
            while myseq not in self.responses:
                cvar.wait()
            response = self.responses[myseq]
            self.debug("_getresponse:%s: thread woke up: response: %s" %
                       (myseq, response))
            del self.responses[myseq]
            del self.cvars[myseq]
            cvar.release()
            return response

    def newseq(self):
        self.nextseq = seq = self.nextseq + 2
        return seq

    def putmessage(self, message):
        self.debug("putmessage:%d:" % message[0])
        try:
            s = dumps(message)
        except pickle.PicklingError:
            print("Cannot pickle:", repr(message), file=sys.__stderr__)
            raise
        s = struct.pack("<i", len(s)) + s
        while len(s) > 0:
            try:
                r, w, x = select.select([], [self.sock], [])
                n = self.sock.send(s[:BUFSIZE])
            except (AttributeError, TypeError):
                raise OSError("socket no longer exists")
            s = s[n:]

    buff = b''
    bufneed = 4
    bufstate = 0 # meaning: 0 => reading count; 1 => reading data

    def pollpacket(self, wait):
        self._stage0()
        if len(self.buff) < self.bufneed:
            r, w, x = select.select([self.sock.fileno()], [], [], wait)
            if len(r) == 0:
                return None
            try:
                s = self.sock.recv(BUFSIZE)
            except OSError:
                raise EOFError
            if len(s) == 0:
                raise EOFError
            self.buff += s
            self._stage0()
        return self._stage1()

    def _stage0(self):
        if self.bufstate == 0 and len(self.buff) >= 4:
            s = self.buff[:4]
            self.buff = self.buff[4:]
            self.bufneed = struct.unpack("<i", s)[0]
            self.bufstate = 1

    def _stage1(self):
        if self.bufstate == 1 and len(self.buff) >= self.bufneed:
            packet = self.buff[:self.bufneed]
            self.buff = self.buff[self.bufneed:]
            self.bufneed = 4
            self.bufstate = 0
            return packet

    def pollmessage(self, wait):
        packet = self.pollpacket(wait)
        if packet is None:
            return None
        try:
            message = pickle.loads(packet)
        except pickle.UnpicklingError:
            print("-----------------------", file=sys.__stderr__)
            print("cannot unpickle packet:", repr(packet), file=sys.__stderr__)
            traceback.print_stack(file=sys.__stderr__)
            print("-----------------------", file=sys.__stderr__)
            raise
        return message

    def pollresponse(self, myseq, wait):
        """Handle messages received on the socket.

        Some messages received may be asynchronous 'call' or 'queue' requests,
        and some may be responses for other threads.

        'call' requests are passed to self.localcall() with the expectation of
        immediate execution, during which time the socket is not serviced.

        'queue' requests are used for tasks (which may block or hang) to be
        processed in a different thread.  These requests are fed into
        request_queue by self.localcall().  Responses to queued requests are
        taken from response_queue and sent across the link with the associated
        sequence numbers.  Messages in the queues are (sequence_number,
        request/response) tuples and code using this module removing messages
        from the request_queue is responsible for returning the correct
        sequence number in the response_queue.

        pollresponse() will loop until a response message with the myseq
        sequence number is received, and will save other responses in
        self.responses and notify the owning thread.

        """
        while True:
            # send queued response if there is one available
            try:
                qmsg = response_queue.get(0)
            except queue.Empty:
                pass
            else:
                seq, response = qmsg
                message = (seq, ('OK', response))
                self.putmessage(message)
            # poll for message on link
            try:
                message = self.pollmessage(wait)
                if message is None:  # socket not ready
                    return None
            except EOFError:
                self.handle_EOF()
                return None
            except AttributeError:
                return None
            seq, resq = message
            how = resq[0]
            self.debug("pollresponse:%d:myseq:%s" % (seq, myseq))
            # process or queue a request
            if how in ("CALL", "QUEUE"):
                self.debug("pollresponse:%d:localcall:call:" % seq)
                response = self.localcall(seq, resq)
                self.debug("pollresponse:%d:localcall:response:%s"
                           % (seq, response))
                if how == "CALL":
                    self.putmessage((seq, response))
                elif how == "QUEUE":
                    # don't acknowledge the 'queue' request!
                    pass
                continue
            # return if completed message transaction
            elif seq == myseq:
                return resq
            # must be a response for a different thread:
            else:
                cv = self.cvars.get(seq, None)
                # response involving unknown sequence number is discarded,
                # probably intended for prior incarnation of server
                if cv is not None:
                    cv.acquire()
                    self.responses[seq] = resq
                    cv.notify()
                    cv.release()
                continue

    def handle_EOF(self):
        "action taken upon link being closed by peer"
        self.EOFhook()
        self.debug("handle_EOF")
        for key in self.cvars:
            cv = self.cvars[key]
            cv.acquire()
            self.responses[key] = ('EOF', None)
            cv.notify()
            cv.release()
        # call our (possibly overridden) exit function
        self.exithook()

    def EOFhook(self):
        "Classes using rpc client/server can override to augment EOF action"
        pass

#----------------- end class SocketIO --------------------

class RemoteObject:
    # Token mix-in class
    pass


def remoteref(obj):
    oid = id(obj)
    objecttable[oid] = obj
    return RemoteProxy(oid)


class RemoteProxy:

    def __init__(self, oid):
        self.oid = oid


class RPCHandler(socketserver.BaseRequestHandler, SocketIO):

    debugging = False
    location = "#S"  # Server

    def __init__(self, sock, addr, svr):
        svr.current_handler = self ## cgt xxx
        SocketIO.__init__(self, sock)
        socketserver.BaseRequestHandler.__init__(self, sock, addr, svr)

    def handle(self):
        "handle() method required by socketserver"
        self.mainloop()

    def get_remote_proxy(self, oid):
        return RPCProxy(self, oid)


class RPCClient(SocketIO):

    debugging = False
    location = "#C"  # Client

    nextseq = 1 # Requests coming from the client are odd numbered

    def __init__(self, address, family=socket.AF_INET, type=socket.SOCK_STREAM):
        self.listening_sock = socket.socket(family, type)
        self.listening_sock.bind(address)
        self.listening_sock.listen(1)

    def accept(self):
        working_sock, address = self.listening_sock.accept()
        if self.debugging:
            print("****** Connection request from ", address, file=sys.__stderr__)
        if address[0] == LOCALHOST:
            SocketIO.__init__(self, working_sock)
        else:
            print("** Invalid host: ", address, file=sys.__stderr__)
            raise OSError

    def get_remote_proxy(self, oid):
        return RPCProxy(self, oid)


class RPCProxy:

    __methods = None
    __attributes = None

    def __init__(self, sockio, oid):
        self.sockio = sockio
        self.oid = oid

    def __getattr__(self, name):
        if self.__methods is None:
            self.__getmethods()
        if self.__methods.get(name):
            return MethodProxy(self.sockio, self.oid, name)
        if self.__attributes is None:
            self.__getattributes()
        if name in self.__attributes:
            value = self.sockio.remotecall(self.oid, '__getattribute__',
                                           (name,), {})
            return value
        else:
            raise AttributeError(name)

    def __getattributes(self):
        self.__attributes = self.sockio.remotecall(self.oid,
                                                "__attributes__", (), {})

    def __getmethods(self):
        self.__methods = self.sockio.remotecall(self.oid,
                                                "__methods__", (), {})

def _getmethods(obj, methods):
    # Helper to get a list of methods from an object
    # Adds names to dictionary argument 'methods'
    for name in dir(obj):
        attr = getattr(obj, name)
        if callable(attr):
            methods[name] = 1
    if isinstance(obj, type):
        for super in obj.__bases__:
            _getmethods(super, methods)

def _getattributes(obj, attributes):
    for name in dir(obj):
        attr = getattr(obj, name)
        if not callable(attr):
            attributes[name] = 1


class MethodProxy:

    def __init__(self, sockio, oid, name):
        self.sockio = sockio
        self.oid = oid
        self.name = name

    def __call__(self, /, *args, **kwargs):
        value = self.sockio.remotecall(self.oid, self.name, args, kwargs)
        return value


# XXX KBK 09Sep03  We need a proper unit test for this module.  Previously
#                  existing test code was removed at Rev 1.27 (r34098).

def displayhook(value):
    """Override standard display hook to use non-locale encoding"""
    if value is None:
        return
    # Set '_' to None to avoid recursion
    builtins._ = None
    text = repr(value)
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        # let's use ascii while utf8-bmp codec doesn't present
        encoding = 'ascii'
        bytes = text.encode(encoding, 'backslashreplace')
        text = bytes.decode(encoding, 'strict')
        sys.stdout.write(text)
    sys.stdout.write("\n")
    builtins._ = value


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_rpc', verbosity=2,)
