#ifndef Py_INTERNAL_CORECONFIG_H
#define Py_INTERNAL_CORECONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* Forward declaration */
struct pyruntimestate;

/* --- PyStatus ----------------------------------------------- */

/* Almost all errors causing Python initialization to fail */
#ifdef _MSC_VER
   /* Visual Studio 2015 doesn't implement C99 __func__ in C */
#  define _PyStatus_GET_FUNC() __FUNCTION__
#else
#  define _PyStatus_GET_FUNC() __func__
#endif

#define _PyStatus_OK() \
    (PyStatus){._type = _PyStatus_TYPE_OK,}
    /* other fields are set to 0 */
#define _PyStatus_ERR(ERR_MSG) \
    (PyStatus){ \
        ._type = _PyStatus_TYPE_ERROR, \
        .func = _PyStatus_GET_FUNC(), \
        .err_msg = (ERR_MSG)}
        /* other fields are set to 0 */
#define _PyStatus_NO_MEMORY() _PyStatus_ERR("memory allocation failed")
#define _PyStatus_EXIT(EXITCODE) \
    (PyStatus){ \
        ._type = _PyStatus_TYPE_EXIT, \
        .exitcode = (EXITCODE)}
#define _PyStatus_IS_ERROR(err) \
    (err._type == _PyStatus_TYPE_ERROR)
#define _PyStatus_IS_EXIT(err) \
    (err._type == _PyStatus_TYPE_EXIT)
#define _PyStatus_EXCEPTION(err) \
    (err._type != _PyStatus_TYPE_OK)
#define _PyStatus_UPDATE_FUNC(err) \
    do { err.func = _PyStatus_GET_FUNC(); } while (0)

PyObject* _PyErr_SetFromPyStatus(PyStatus status);

/* --- PyWideStringList ------------------------------------------------ */

#define _PyWideStringList_INIT (PyWideStringList){.length = 0, .items = NULL}

#ifndef NDEBUG
PyAPI_FUNC(int) _PyWideStringList_CheckConsistency(const PyWideStringList *list);
#endif
PyAPI_FUNC(void) _PyWideStringList_Clear(PyWideStringList *list);
PyAPI_FUNC(int) _PyWideStringList_Copy(PyWideStringList *list,
    const PyWideStringList *list2);
PyAPI_FUNC(PyStatus) _PyWideStringList_Extend(PyWideStringList *list,
    const PyWideStringList *list2);
PyAPI_FUNC(PyObject*) _PyWideStringList_AsList(const PyWideStringList *list);


/* --- _PyArgv ---------------------------------------------------- */

typedef struct _PyArgv {
    Py_ssize_t argc;
    int use_bytes_argv;
    char * const *bytes_argv;
    wchar_t * const *wchar_argv;
} _PyArgv;

PyAPI_FUNC(PyStatus) _PyArgv_AsWstrList(const _PyArgv *args,
    PyWideStringList *list);


/* --- Helper functions ------------------------------------------- */

PyAPI_FUNC(int) _Py_str_to_int(
    const char *str,
    int *result);
PyAPI_FUNC(const wchar_t*) _Py_get_xoption(
    const PyWideStringList *xoptions,
    const wchar_t *name);
PyAPI_FUNC(const char*) _Py_GetEnv(
    int use_environment,
    const char *name);
PyAPI_FUNC(void) _Py_get_env_flag(
    int use_environment,
    int *flag,
    const char *name);

/* Py_GetArgcArgv() helper */
PyAPI_FUNC(void) _Py_ClearArgcArgv(void);


/* --- _PyPreCmdline ------------------------------------------------- */

typedef struct {
    PyWideStringList argv;
    PyWideStringList xoptions;     /* "-X value" option */
    int isolated;             /* -I option */
    int use_environment;      /* -E option */
    int dev_mode;             /* -X dev and PYTHONDEVMODE */
    int warn_default_encoding;     /* -X warn_default_encoding and PYTHONWARNDEFAULTENCODING */
} _PyPreCmdline;

#define _PyPreCmdline_INIT \
    (_PyPreCmdline){ \
        .use_environment = -1, \
        .isolated = -1, \
        .dev_mode = -1}
/* Note: _PyPreCmdline_INIT sets other fields to 0/NULL */

extern void _PyPreCmdline_Clear(_PyPreCmdline *cmdline);
extern PyStatus _PyPreCmdline_SetArgv(_PyPreCmdline *cmdline,
    const _PyArgv *args);
extern PyStatus _PyPreCmdline_SetConfig(
    const _PyPreCmdline *cmdline,
    PyConfig *config);
extern PyStatus _PyPreCmdline_Read(_PyPreCmdline *cmdline,
    const PyPreConfig *preconfig);


/* --- PyPreConfig ----------------------------------------------- */

PyAPI_FUNC(void) _PyPreConfig_InitCompatConfig(PyPreConfig *preconfig);
extern void _PyPreConfig_InitFromConfig(
    PyPreConfig *preconfig,
    const PyConfig *config);
extern PyStatus _PyPreConfig_InitFromPreConfig(
    PyPreConfig *preconfig,
    const PyPreConfig *config2);
extern PyObject* _PyPreConfig_AsDict(const PyPreConfig *preconfig);
extern void _PyPreConfig_GetConfig(PyPreConfig *preconfig,
    const PyConfig *config);
extern PyStatus _PyPreConfig_Read(PyPreConfig *preconfig,
    const _PyArgv *args);
extern PyStatus _PyPreConfig_Write(const PyPreConfig *preconfig);


/* --- PyConfig ---------------------------------------------- */

typedef enum {
    /* Py_Initialize() API: backward compatibility with Python 3.6 and 3.7 */
    _PyConfig_INIT_COMPAT = 1,
    _PyConfig_INIT_PYTHON = 2,
    _PyConfig_INIT_ISOLATED = 3
} _PyConfigInitEnum;

PyAPI_FUNC(void) _PyConfig_InitCompatConfig(PyConfig *config);
extern PyStatus _PyConfig_Copy(
    PyConfig *config,
    const PyConfig *config2);
extern PyStatus _PyConfig_InitPathConfig(
    PyConfig *config,
    int compute_path_config);
extern PyStatus _PyConfig_InitImportConfig(PyConfig *config);
extern PyStatus _PyConfig_Read(PyConfig *config, int compute_path_config);
extern PyStatus _PyConfig_Write(const PyConfig *config,
    struct pyruntimestate *runtime);
extern PyStatus _PyConfig_SetPyArgv(
    PyConfig *config,
    const _PyArgv *args);

PyAPI_FUNC(PyObject*) _PyConfig_AsDict(const PyConfig *config);
PyAPI_FUNC(int) _PyConfig_FromDict(PyConfig *config, PyObject *dict);

extern void _Py_DumpPathConfig(PyThreadState *tstate);

PyAPI_FUNC(PyObject*) _Py_Get_Getpath_CodeObject(void);

extern int _Py_global_config_int_max_str_digits;


/* --- Function used for testing ---------------------------------- */

PyAPI_FUNC(PyObject*) _Py_GetConfigsAsDict(void);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_CORECONFIG_H */
