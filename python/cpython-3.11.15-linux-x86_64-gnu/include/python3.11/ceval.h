/* Interface to random parts in ceval.c */

#ifndef Py_CEVAL_H
#define Py_CEVAL_H
#ifdef __cplusplus
extern "C" {
#endif


PyAPI_FUNC(PyObject *) PyEval_EvalCode(PyObject *, PyObject *, PyObject *);

PyAPI_FUNC(PyObject *) PyEval_EvalCodeEx(PyObject *co,
                                         PyObject *globals,
                                         PyObject *locals,
                                         PyObject *const *args, int argc,
                                         PyObject *const *kwds, int kwdc,
                                         PyObject *const *defs, int defc,
                                         PyObject *kwdefs, PyObject *closure);

/* PyEval_CallObjectWithKeywords(), PyEval_CallObject(), PyEval_CallFunction
 * and PyEval_CallMethod are deprecated. Since they are officially part of the
 * stable ABI (PEP 384), they must be kept for backward compatibility.
 * PyObject_Call(), PyObject_CallFunction() and PyObject_CallMethod() are
 * recommended to call a callable object.
 */

Py_DEPRECATED(3.9) PyAPI_FUNC(PyObject *) PyEval_CallObjectWithKeywords(
    PyObject *callable,
    PyObject *args,
    PyObject *kwargs);

/* Deprecated since PyEval_CallObjectWithKeywords is deprecated */
#define PyEval_CallObject(callable, arg) \
    PyEval_CallObjectWithKeywords(callable, arg, (PyObject *)NULL)

Py_DEPRECATED(3.9) PyAPI_FUNC(PyObject *) PyEval_CallFunction(
    PyObject *callable, const char *format, ...);
Py_DEPRECATED(3.9) PyAPI_FUNC(PyObject *) PyEval_CallMethod(
    PyObject *obj, const char *name, const char *format, ...);

PyAPI_FUNC(PyObject *) PyEval_GetBuiltins(void);
PyAPI_FUNC(PyObject *) PyEval_GetGlobals(void);
PyAPI_FUNC(PyObject *) PyEval_GetLocals(void);
PyAPI_FUNC(PyFrameObject *) PyEval_GetFrame(void);

PyAPI_FUNC(int) Py_AddPendingCall(int (*func)(void *), void *arg);
PyAPI_FUNC(int) Py_MakePendingCalls(void);

/* Protection against deeply nested recursive calls

   In Python 3.0, this protection has two levels:
   * normal anti-recursion protection is triggered when the recursion level
     exceeds the current recursion limit. It raises a RecursionError, and sets
     the "overflowed" flag in the thread state structure. This flag
     temporarily *disables* the normal protection; this allows cleanup code
     to potentially outgrow the recursion limit while processing the
     RecursionError.
   * "last chance" anti-recursion protection is triggered when the recursion
     level exceeds "current recursion limit + 50". By construction, this
     protection can only be triggered when the "overflowed" flag is set. It
     means the cleanup code has itself gone into an infinite loop, or the
     RecursionError has been mistakingly ignored. When this protection is
     triggered, the interpreter aborts with a Fatal Error.

   In addition, the "overflowed" flag is automatically reset when the
   recursion level drops below "current recursion limit - 50". This heuristic
   is meant to ensure that the normal anti-recursion protection doesn't get
   disabled too long.

   Please note: this scheme has its own limitations. See:
   http://mail.python.org/pipermail/python-dev/2008-August/082106.html
   for some observations.
*/
PyAPI_FUNC(void) Py_SetRecursionLimit(int);
PyAPI_FUNC(int) Py_GetRecursionLimit(void);

PyAPI_FUNC(int) Py_EnterRecursiveCall(const char *where);
PyAPI_FUNC(void) Py_LeaveRecursiveCall(void);

PyAPI_FUNC(const char *) PyEval_GetFuncName(PyObject *);
PyAPI_FUNC(const char *) PyEval_GetFuncDesc(PyObject *);

PyAPI_FUNC(PyObject *) PyEval_EvalFrame(PyFrameObject *);
PyAPI_FUNC(PyObject *) PyEval_EvalFrameEx(PyFrameObject *f, int exc);

/* Interface for threads.

   A module that plans to do a blocking system call (or something else
   that lasts a long time and doesn't touch Python data) can allow other
   threads to run as follows:

    ...preparations here...
    Py_BEGIN_ALLOW_THREADS
    ...blocking system call here...
    Py_END_ALLOW_THREADS
    ...interpret result here...

   The Py_BEGIN_ALLOW_THREADS/Py_END_ALLOW_THREADS pair expands to a
   {}-surrounded block.
   To leave the block in the middle (e.g., with return), you must insert
   a line containing Py_BLOCK_THREADS before the return, e.g.

    if (...premature_exit...) {
        Py_BLOCK_THREADS
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

   An alternative is:

    Py_BLOCK_THREADS
    if (...premature_exit...) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }
    Py_UNBLOCK_THREADS

   For convenience, that the value of 'errno' is restored across
   Py_END_ALLOW_THREADS and Py_BLOCK_THREADS.

   WARNING: NEVER NEST CALLS TO Py_BEGIN_ALLOW_THREADS AND
   Py_END_ALLOW_THREADS!!!

   Note that not yet all candidates have been converted to use this
   mechanism!
*/

PyAPI_FUNC(PyThreadState *) PyEval_SaveThread(void);
PyAPI_FUNC(void) PyEval_RestoreThread(PyThreadState *);

Py_DEPRECATED(3.9) PyAPI_FUNC(int) PyEval_ThreadsInitialized(void);
Py_DEPRECATED(3.9) PyAPI_FUNC(void) PyEval_InitThreads(void);
/* PyEval_AcquireLock() and PyEval_ReleaseLock() are part of stable ABI.
 * They will be removed from this header file in the future version.
 * But they will be remained in ABI until Python 4.0.
 */
Py_DEPRECATED(3.2) PyAPI_FUNC(void) PyEval_AcquireLock(void);
Py_DEPRECATED(3.2) PyAPI_FUNC(void) PyEval_ReleaseLock(void);
PyAPI_FUNC(void) PyEval_AcquireThread(PyThreadState *tstate);
PyAPI_FUNC(void) PyEval_ReleaseThread(PyThreadState *tstate);

#define Py_BEGIN_ALLOW_THREADS { \
                        PyThreadState *_save; \
                        _save = PyEval_SaveThread();
#define Py_BLOCK_THREADS        PyEval_RestoreThread(_save);
#define Py_UNBLOCK_THREADS      _save = PyEval_SaveThread();
#define Py_END_ALLOW_THREADS    PyEval_RestoreThread(_save); \
                 }

/* Masks and values used by FORMAT_VALUE opcode. */
#define FVC_MASK      0x3
#define FVC_NONE      0x0
#define FVC_STR       0x1
#define FVC_REPR      0x2
#define FVC_ASCII     0x3
#define FVS_MASK      0x4
#define FVS_HAVE_SPEC 0x4

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_CEVAL_H
#  include "cpython/ceval.h"
#  undef Py_CPYTHON_CEVAL_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_CEVAL_H */
