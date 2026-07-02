#ifndef Py_INTERNAL_PYSTATE_H
#define Py_INTERNAL_PYSTATE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_runtime.h"   /* PyRuntimeState */


/* Check if the current thread is the main thread.
   Use _Py_IsMainInterpreter() to check if it's the main interpreter. */
static inline int
_Py_IsMainThread(void)
{
    unsigned long thread = PyThread_get_thread_ident();
    return (thread == _PyRuntime.main_thread);
}


static inline PyInterpreterState *
_PyInterpreterState_Main(void)
{
    return _PyRuntime.interpreters.main;
}

static inline int
_Py_IsMainInterpreter(PyInterpreterState *interp)
{
    return (interp == _PyInterpreterState_Main());
}


static inline const PyConfig *
_Py_GetMainConfig(void)
{
    PyInterpreterState *interp = _PyInterpreterState_Main();
    if (interp == NULL) {
        return NULL;
    }
    return _PyInterpreterState_GetConfig(interp);
}


/* Only handle signals on the main thread of the main interpreter. */
static inline int
_Py_ThreadCanHandleSignals(PyInterpreterState *interp)
{
    return (_Py_IsMainThread() && _Py_IsMainInterpreter(interp));
}


/* Only execute pending calls on the main thread. */
static inline int
_Py_ThreadCanHandlePendingCalls(void)
{
    return _Py_IsMainThread();
}


#ifndef NDEBUG
extern int _PyThreadState_CheckConsistency(PyThreadState *tstate);
#endif

int _PyThreadState_MustExit(PyThreadState *tstate);

/* Variable and macro for in-line access to current thread
   and interpreter state */

static inline PyThreadState*
_PyRuntimeState_GetThreadState(_PyRuntimeState *runtime)
{
    return (PyThreadState*)_Py_atomic_load_relaxed(&runtime->gilstate.tstate_current);
}

/* Get the current Python thread state.

   Efficient macro reading directly the 'gilstate.tstate_current' atomic
   variable. The macro is unsafe: it does not check for error and it can
   return NULL.

   The caller must hold the GIL.

   See also PyThreadState_Get() and _PyThreadState_UncheckedGet(). */
static inline PyThreadState*
_PyThreadState_GET(void)
{
    return _PyRuntimeState_GetThreadState(&_PyRuntime);
}

PyAPI_FUNC(void) _Py_NO_RETURN _Py_FatalError_TstateNULL(const char *func);

static inline void
_Py_EnsureFuncTstateNotNULL(const char *func, PyThreadState *tstate)
{
    if (tstate == NULL) {
        _Py_FatalError_TstateNULL(func);
    }
}

// Call Py_FatalError() if tstate is NULL
#define _Py_EnsureTstateNotNULL(tstate) \
    _Py_EnsureFuncTstateNotNULL(__func__, tstate)


/* Get the current interpreter state.

   The macro is unsafe: it does not check for error and it can return NULL.

   The caller must hold the GIL.

   See also _PyInterpreterState_Get()
   and _PyGILState_GetInterpreterStateUnsafe(). */
static inline PyInterpreterState* _PyInterpreterState_GET(void) {
    PyThreadState *tstate = _PyThreadState_GET();
#ifdef Py_DEBUG
    _Py_EnsureTstateNotNULL(tstate);
#endif
    return tstate->interp;
}


// PyThreadState functions

PyAPI_FUNC(void) _PyThreadState_SetCurrent(PyThreadState *tstate);
// We keep this around exclusively for stable ABI compatibility.
PyAPI_FUNC(void) _PyThreadState_Init(
    PyThreadState *tstate);
PyAPI_FUNC(void) _PyThreadState_DeleteExcept(
    _PyRuntimeState *runtime,
    PyThreadState *tstate);


static inline void
_PyThreadState_UpdateTracingState(PyThreadState *tstate)
{
    bool use_tracing =
        (tstate->tracing == 0) &&
        (tstate->c_tracefunc != NULL || tstate->c_profilefunc != NULL);
    tstate->cframe->use_tracing = (use_tracing ? 255 : 0);
}


/* Other */

PyAPI_FUNC(PyThreadState *) _PyThreadState_Swap(
    struct _gilstate_runtime_state *gilstate,
    PyThreadState *newts);

PyAPI_FUNC(PyStatus) _PyInterpreterState_Enable(_PyRuntimeState *runtime);

#ifdef HAVE_FORK
extern PyStatus _PyInterpreterState_DeleteExceptMain(_PyRuntimeState *runtime);
extern PyStatus _PyGILState_Reinit(_PyRuntimeState *runtime);
extern void _PySignal_AfterFork(void);
#endif


PyAPI_FUNC(int) _PyState_AddModule(
    PyThreadState *tstate,
    PyObject* module,
    PyModuleDef* def);


PyAPI_FUNC(int) _PyOS_InterruptOccurred(PyThreadState *tstate);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PYSTATE_H */
