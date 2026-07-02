/* Thread and interpreter state structures and their interfaces */


#ifndef Py_PYSTATE_H
#define Py_PYSTATE_H
#ifdef __cplusplus
extern "C" {
#endif

/* This limitation is for performance and simplicity. If needed it can be
removed (with effort). */
#define MAX_CO_EXTRA_USERS 255

PyAPI_FUNC(PyInterpreterState *) PyInterpreterState_New(void);
PyAPI_FUNC(void) PyInterpreterState_Clear(PyInterpreterState *);
PyAPI_FUNC(void) PyInterpreterState_Delete(PyInterpreterState *);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03090000
/* New in 3.9 */
/* Get the current interpreter state.

   Issue a fatal error if there no current Python thread state or no current
   interpreter. It cannot return NULL.

   The caller must hold the GIL. */
PyAPI_FUNC(PyInterpreterState *) PyInterpreterState_Get(void);
#endif

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03080000
/* New in 3.8 */
PyAPI_FUNC(PyObject *) PyInterpreterState_GetDict(PyInterpreterState *);
#endif

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03070000
/* New in 3.7 */
PyAPI_FUNC(int64_t) PyInterpreterState_GetID(PyInterpreterState *);
#endif
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03030000

/* State unique per thread */

/* New in 3.3 */
PyAPI_FUNC(int) PyState_AddModule(PyObject*, PyModuleDef*);
PyAPI_FUNC(int) PyState_RemoveModule(PyModuleDef*);
#endif
PyAPI_FUNC(PyObject*) PyState_FindModule(PyModuleDef*);

PyAPI_FUNC(PyThreadState *) PyThreadState_New(PyInterpreterState *);
PyAPI_FUNC(void) PyThreadState_Clear(PyThreadState *);
PyAPI_FUNC(void) PyThreadState_Delete(PyThreadState *);

/* Get the current thread state.

   When the current thread state is NULL, this issues a fatal error (so that
   the caller needn't check for NULL).

   The caller must hold the GIL.

   See also _PyThreadState_UncheckedGet() and _PyThreadState_GET(). */
PyAPI_FUNC(PyThreadState *) PyThreadState_Get(void);

// Alias to PyThreadState_Get()
#define PyThreadState_GET() PyThreadState_Get()

PyAPI_FUNC(PyThreadState *) PyThreadState_Swap(PyThreadState *);
PyAPI_FUNC(PyObject *) PyThreadState_GetDict(void);
PyAPI_FUNC(int) PyThreadState_SetAsyncExc(unsigned long, PyObject *);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03090000
/* New in 3.9 */
PyAPI_FUNC(PyInterpreterState*) PyThreadState_GetInterpreter(PyThreadState *tstate);
PyAPI_FUNC(PyFrameObject*) PyThreadState_GetFrame(PyThreadState *tstate);
PyAPI_FUNC(uint64_t) PyThreadState_GetID(PyThreadState *tstate);
#endif

typedef
    enum {PyGILState_LOCKED, PyGILState_UNLOCKED}
        PyGILState_STATE;


/* Ensure that the current thread is ready to call the Python
   C API, regardless of the current state of Python, or of its
   thread lock.  This may be called as many times as desired
   by a thread so long as each call is matched with a call to
   PyGILState_Release().  In general, other thread-state APIs may
   be used between _Ensure() and _Release() calls, so long as the
   thread-state is restored to its previous state before the Release().
   For example, normal use of the Py_BEGIN_ALLOW_THREADS/
   Py_END_ALLOW_THREADS macros are acceptable.

   The return value is an opaque "handle" to the thread state when
   PyGILState_Ensure() was called, and must be passed to
   PyGILState_Release() to ensure Python is left in the same state. Even
   though recursive calls are allowed, these handles can *not* be shared -
   each unique call to PyGILState_Ensure must save the handle for its
   call to PyGILState_Release.

   When the function returns, the current thread will hold the GIL.

   Failure is a fatal error.
*/
PyAPI_FUNC(PyGILState_STATE) PyGILState_Ensure(void);

/* Release any resources previously acquired.  After this call, Python's
   state will be the same as it was prior to the corresponding
   PyGILState_Ensure() call (but generally this state will be unknown to
   the caller, hence the use of the GILState API.)

   Every call to PyGILState_Ensure must be matched by a call to
   PyGILState_Release on the same thread.
*/
PyAPI_FUNC(void) PyGILState_Release(PyGILState_STATE);

/* Helper/diagnostic function - get the current thread state for
   this thread.  May return NULL if no GILState API has been used
   on the current thread.  Note that the main thread always has such a
   thread-state, even if no auto-thread-state call has been made
   on the main thread.
*/
PyAPI_FUNC(PyThreadState *) PyGILState_GetThisThreadState(void);


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_PYSTATE_H
#  include "cpython/pystate.h"
#  undef Py_CPYTHON_PYSTATE_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_PYSTATE_H */
