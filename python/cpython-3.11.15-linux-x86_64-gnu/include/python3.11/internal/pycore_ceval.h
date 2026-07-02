#ifndef Py_INTERNAL_CEVAL_H
#define Py_INTERNAL_CEVAL_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* Forward declarations */
struct pyruntimestate;
struct _ceval_runtime_state;

/* WASI has limited call stack. Python's recursion limit depends on code
   layout, optimization, and WASI runtime. Wasmtime can handle about 700-750
   recursions, sometimes less. 600 is a more conservative limit. */
#ifndef Py_DEFAULT_RECURSION_LIMIT
#  ifdef __wasi__
#    define Py_DEFAULT_RECURSION_LIMIT 600
#  else
#    define Py_DEFAULT_RECURSION_LIMIT 1000
#  endif
#endif

#include "pycore_interp.h"        // PyInterpreterState.eval_frame
#include "pycore_pystate.h"       // _PyThreadState_GET()


extern void _Py_FinishPendingCalls(PyThreadState *tstate);
extern void _PyEval_InitRuntimeState(struct _ceval_runtime_state *);
extern void _PyEval_InitState(struct _ceval_state *, PyThread_type_lock);
extern void _PyEval_FiniState(struct _ceval_state *ceval);
PyAPI_FUNC(void) _PyEval_SignalReceived(PyInterpreterState *interp);
PyAPI_FUNC(int) _PyEval_AddPendingCall(
    PyInterpreterState *interp,
    int (*func)(void *),
    void *arg);
PyAPI_FUNC(void) _PyEval_SignalAsyncExc(PyInterpreterState *interp);
#ifdef HAVE_FORK
extern PyStatus _PyEval_ReInitThreads(PyThreadState *tstate);
#endif

// Used by sys.call_tracing()
extern PyObject* _PyEval_CallTracing(PyObject *func, PyObject *args);

// Used by sys.get_asyncgen_hooks()
extern PyObject* _PyEval_GetAsyncGenFirstiter(void);
extern PyObject* _PyEval_GetAsyncGenFinalizer(void);

// Used by sys.set_asyncgen_hooks()
extern int _PyEval_SetAsyncGenFirstiter(PyObject *);
extern int _PyEval_SetAsyncGenFinalizer(PyObject *);

// Used by sys.get_coroutine_origin_tracking_depth()
// and sys.set_coroutine_origin_tracking_depth()
extern int _PyEval_GetCoroutineOriginTrackingDepth(void);
extern int _PyEval_SetCoroutineOriginTrackingDepth(int depth);

extern void _PyEval_Fini(void);


extern PyObject* _PyEval_GetBuiltins(PyThreadState *tstate);
extern PyObject* _PyEval_BuiltinsFromGlobals(
    PyThreadState *tstate,
    PyObject *globals);


static inline PyObject*
_PyEval_EvalFrame(PyThreadState *tstate, struct _PyInterpreterFrame *frame, int throwflag)
{
    if (tstate->interp->eval_frame == NULL) {
        return _PyEval_EvalFrameDefault(tstate, frame, throwflag);
    }
    return tstate->interp->eval_frame(tstate, frame, throwflag);
}

extern PyObject*
_PyEval_Vector(PyThreadState *tstate,
            PyFunctionObject *func, PyObject *locals,
            PyObject* const* args, size_t argcount,
            PyObject *kwnames);

extern int _PyEval_ThreadsInitialized(struct pyruntimestate *runtime);
extern PyStatus _PyEval_InitGIL(PyThreadState *tstate);
extern void _PyEval_FiniGIL(PyInterpreterState *interp);

extern void _PyEval_ReleaseLock(PyThreadState *tstate);

extern void _PyEval_DeactivateOpCache(void);


/* --- _Py_EnterRecursiveCall() ----------------------------------------- */

#ifdef USE_STACKCHECK
/* With USE_STACKCHECK macro defined, trigger stack checks in
   _Py_CheckRecursiveCall() on every 64th call to _Py_EnterRecursiveCall. */
static inline int _Py_MakeRecCheck(PyThreadState *tstate)  {
    return (tstate->recursion_remaining-- <= 0
            || (tstate->recursion_remaining & 63) == 0);
}
#else
static inline int _Py_MakeRecCheck(PyThreadState *tstate) {
    return tstate->recursion_remaining-- <= 0;
}
#endif

PyAPI_FUNC(int) _Py_CheckRecursiveCall(
    PyThreadState *tstate,
    const char *where);

static inline int _Py_EnterRecursiveCallTstate(PyThreadState *tstate,
                                               const char *where) {
    return (_Py_MakeRecCheck(tstate) && _Py_CheckRecursiveCall(tstate, where));
}

static inline int _Py_EnterRecursiveCall(const char *where) {
    PyThreadState *tstate = _PyThreadState_GET();
    return _Py_EnterRecursiveCallTstate(tstate, where);
}

static inline void _Py_LeaveRecursiveCallTstate(PyThreadState *tstate)  {
    tstate->recursion_remaining++;
}

static inline void _Py_LeaveRecursiveCall(void)  {
    PyThreadState *tstate = _PyThreadState_GET();
    _Py_LeaveRecursiveCallTstate(tstate);
}

extern struct _PyInterpreterFrame* _PyEval_GetFrame(void);

extern PyObject* _Py_MakeCoro(PyFunctionObject *func);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_CEVAL_H */
