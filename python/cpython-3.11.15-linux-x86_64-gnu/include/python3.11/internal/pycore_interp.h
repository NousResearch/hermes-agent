#ifndef Py_INTERNAL_INTERP_H
#define Py_INTERNAL_INTERP_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include <stdbool.h>

#include "pycore_atomic.h"        // _Py_atomic_address
#include "pycore_ast_state.h"     // struct ast_state
#include "pycore_code.h"          // struct callable_cache
#include "pycore_context.h"       // struct _Py_context_state
#include "pycore_dict.h"          // struct _Py_dict_state
#include "pycore_exceptions.h"    // struct _Py_exc_state
#include "pycore_floatobject.h"   // struct _Py_float_state
#include "pycore_genobject.h"     // struct _Py_async_gen_state
#include "pycore_gil.h"           // struct _gil_runtime_state
#include "pycore_gc.h"            // struct _gc_runtime_state
#include "pycore_list.h"          // struct _Py_list_state
#include "pycore_tuple.h"         // struct _Py_tuple_state
#include "pycore_typeobject.h"    // struct type_cache
#include "pycore_unicodeobject.h" // struct _Py_unicode_state
#include "pycore_warnings.h"      // struct _warnings_runtime_state

struct _pending_calls {
    PyThread_type_lock lock;
    /* Request for running pending calls. */
    _Py_atomic_int calls_to_do;
    /* Request for looking at the `async_exc` field of the current
       thread state.
       Guarded by the GIL. */
    int async_exc;
#define NPENDINGCALLS 32
    struct {
        int (*func)(void *);
        void *arg;
    } calls[NPENDINGCALLS];
    int first;
    int last;
};

struct _ceval_state {
    int recursion_limit;
    /* This single variable consolidates all requests to break out of
       the fast path in the eval loop. */
    _Py_atomic_int eval_breaker;
    /* Request for dropping the GIL */
    _Py_atomic_int gil_drop_request;
    struct _pending_calls pending;
};


// atexit state
typedef struct {
    PyObject *func;
    PyObject *args;
    PyObject *kwargs;
} atexit_callback;

struct atexit_state {
    atexit_callback **callbacks;
    int ncallbacks;
    int callback_len;
};


/* interpreter state */

/* PyInterpreterState holds the global state for one of the runtime's
   interpreters.  Typically the initial (main) interpreter is the only one.

   The PyInterpreterState typedef is in Include/pystate.h.
   */
struct _is {

    PyInterpreterState *next;

    struct pythreads {
        uint64_t next_unique_id;
        /* The linked list of threads, newest first. */
        PyThreadState *head;
        /* Used in Modules/_threadmodule.c. */
        long count;
        /* Support for runtime thread stack size tuning.
           A value of 0 means using the platform's default stack size
           or the size specified by the THREAD_STACK_SIZE macro. */
        /* Used in Python/thread.c. */
        size_t stacksize;
    } threads;

    /* Reference to the _PyRuntime global variable. This field exists
       to not have to pass runtime in addition to tstate to a function.
       Get runtime from tstate: tstate->interp->runtime. */
    struct pyruntimestate *runtime;

    int64_t id;
    int64_t id_refcount;
    int requires_idref;
    PyThread_type_lock id_mutex;

    /* Has been initialized to a safe state.

       In order to be effective, this must be set to 0 during or right
       after allocation. */
    int _initialized;
    int finalizing;

    /* Was this interpreter statically allocated? */
    bool _static;

    struct _ceval_state ceval;
    struct _gc_runtime_state gc;

    // sys.modules dictionary
    PyObject *modules;
    PyObject *modules_by_index;
    // Dictionary of the sys module
    PyObject *sysdict;
    // Dictionary of the builtins module
    PyObject *builtins;
    // importlib module
    PyObject *importlib;
    // override for config->use_frozen_modules (for tests)
    // (-1: "off", 1: "on", 0: no override)
    int override_frozen_modules;

    PyObject *codec_search_path;
    PyObject *codec_search_cache;
    PyObject *codec_error_registry;
    int codecs_initialized;

    PyConfig config;
#ifdef HAVE_DLOPEN
    int dlopenflags;
#endif

    PyObject *dict;  /* Stores per-interpreter state */

    PyObject *builtins_copy;
    PyObject *import_func;
    // Initialized to _PyEval_EvalFrameDefault().
    _PyFrameEvalFunction eval_frame;

    Py_ssize_t co_extra_user_count;
    freefunc co_extra_freefuncs[MAX_CO_EXTRA_USERS];

#ifdef HAVE_FORK
    PyObject *before_forkers;
    PyObject *after_forkers_parent;
    PyObject *after_forkers_child;
#endif

    struct _warnings_runtime_state warnings;
    struct atexit_state atexit;

    PyObject *audit_hooks;

    struct _Py_unicode_state unicode;
    struct _Py_float_state float_state;
    /* Using a cache is very effective since typically only a single slice is
       created and then deleted again. */
    PySliceObject *slice_cache;

    struct _Py_tuple_state tuple;
    struct _Py_list_state list;
    struct _Py_dict_state dict_state;
    struct _Py_async_gen_state async_gen;
    struct _Py_context_state context;
    struct _Py_exc_state exc_state;

    struct ast_state ast;
    struct type_cache type_cache;
    struct callable_cache callable_cache;

    int int_max_str_digits;

    /* The following fields are here to avoid allocation during init.
       The data is exposed through PyInterpreterState pointer fields.
       These fields should not be accessed directly outside of init.

       All other PyInterpreterState pointer fields are populated when
       needed and default to NULL.

       For now there are some exceptions to that rule, which require
       allocation during init.  These will be addressed on a case-by-case
       basis.  Also see _PyRuntimeState regarding the various mutex fields.
       */

    /* the initial PyInterpreterState.threads.head */
    PyThreadState _initial_thread;
};


/* other API */

extern void _PyInterpreterState_ClearModules(PyInterpreterState *interp);
extern void _PyInterpreterState_Clear(PyThreadState *tstate);


/* cross-interpreter data registry */

/* For now we use a global registry of shareable classes.  An
   alternative would be to add a tp_* slot for a class's
   crossinterpdatafunc. It would be simpler and more efficient. */

struct _xidregitem;

struct _xidregitem {
    PyTypeObject *cls;
    crossinterpdatafunc getdata;
    struct _xidregitem *next;
};

PyAPI_FUNC(PyInterpreterState*) _PyInterpreterState_LookUpID(int64_t);

PyAPI_FUNC(int) _PyInterpreterState_IDInitref(PyInterpreterState *);
PyAPI_FUNC(int) _PyInterpreterState_IDIncref(PyInterpreterState *);
PyAPI_FUNC(void) _PyInterpreterState_IDDecref(PyInterpreterState *);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_INTERP_H */
